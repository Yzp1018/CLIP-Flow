import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (TrainingImages, ValidationImages, TestImages, manifest,
                   disjoint, fingerprint, labelled_images, seed_worker)
from .metrics import calibrate_threshold, evaluate_scores
from .network import CLIPFlow, likelihood_loss
from .settings import Settings, BENCHMARKS, EMERGING


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_device(requested):
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def loader(dataset, cfg, shuffle=False):
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.workers, worker_init_fn=seed_worker,
                      generator=generator, drop_last=False)


def save_checkpoint(path, model, cfg, train_paths, epochs):
    # CLIP remains frozen and is obtained from its checksum-verified upstream URL.
    value = {'format': 'clipflow-consolidated-v1', 'settings': cfg.to_dict(),
             'projection': model.projection.state_dict(), 'flow': model.flow.state_dict(),
             'train_paths': list(map(str, train_paths)), 'train_membership': fingerprint(train_paths),
             'completed_epochs': epochs, 'backbone': 'CLIP ViT-L/14'}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device='auto', cache_dir=None, encoder=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'clipflow-consolidated-v1':
        raise ValueError('Unsupported checkpoint format: use import-legacy with explicit settings and training manifest first')
    cfg = Settings(**checkpoint['settings']).validate()
    model = CLIPFlow(cfg, encoder=encoder, cache_dir=cache_dir)
    model.projection.load_state_dict(checkpoint['projection'], strict=True)
    model.flow.load_state_dict(checkpoint['flow'], strict=True)
    model.to(select_device(device)).eval()
    return model, cfg, checkpoint


def train(cfg, train_manifest, output, device='auto', cache_dir=None, encoder=None, allow_other_count=False):
    cfg.validate()
    if (Path(output) / 'last.pt').exists():
        raise FileExistsError('Use a fresh output directory to preserve existing training results')
    paths = manifest(train_manifest)
    if not allow_other_count and len(paths) != 7200:
        raise ValueError('Paper configuration requires 7200 natural training images; use --allow-other-count for a non-paper run')
    setup_seed(cfg.seed)
    model = CLIPFlow(cfg, encoder=encoder, cache_dir=cache_dir).to(select_device(device))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    batches = loader(TrainingImages(paths, cfg), cfg, shuffle=True)
    output = Path(output)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for image, label in batches:
            image, label = image.to(next(model.parameters()).device), label.to(next(model.parameters()).device)
            logp = model(image)
            loss = likelihood_loss(logp, label, cfg.mode)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(image)
            count += len(image)
        history.append({'epoch': epoch + 1, 'mean_loss': total / count})
        print(json.dumps(history[-1]), flush=True)
        save_checkpoint(output / 'last.pt', model, cfg, paths, epoch + 1)
        write_json(output / 'training.json', {'settings': cfg.to_dict(), 'train_membership': fingerprint(paths),
                   'natural_images': len(paths), 'examples_per_epoch': len(batches.dataset),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'history': history, 'paper_count': len(paths) == 7200})
    return model, history


def predict(model, dataset, cfg):
    model.eval()
    scores, labels = [], []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for image, label in loader(dataset, cfg):
            scores.extend(model.score(image.to(device)).cpu().double().tolist())
            labels.extend(label.tolist())
    return np.asarray(labels), np.asarray(scores)


def calibrate(checkpoint_path, validation_manifest, output, device='auto', cache_dir=None, encoder=None):
    model, cfg, state = load_checkpoint(checkpoint_path, device, cache_dir, encoder)
    paths = manifest(validation_manifest)
    disjoint(state['train_paths'], paths)
    labels, scores = predict(model, ValidationImages(paths, cfg), cfg)
    threshold, quality = calibrate_threshold(labels, scores)
    result = {'threshold': threshold, 'validation_balanced_accuracy': quality,
              'checkpoint_sha256': file_hash(checkpoint_path),
              'validation_paths': list(map(str, paths)), 'validation_membership': fingerprint(paths),
              'validation_natural_count': len(paths), 'perturbation': 'fftshift annulus [30,100] pixels',
              'decision': 'fake if score >= threshold', 'selection': 'maximum balanced validation accuracy'}
    write_json(output, result)
    return result


def evaluate(checkpoint_path, root, output, calibration=None, suite='benchmark', datasets=None,
             mapping=None, device='auto', cache_dir=None, encoder=None, allow_other_count=False):
    model, cfg, state = load_checkpoint(checkpoint_path, device, cache_dir, encoder)
    selected = list(datasets or (BENCHMARKS if suite == 'benchmark' else EMERGING))
    expected = BENCHMARKS if suite == 'benchmark' else EMERGING
    if not selected or any(n not in expected for n in selected) or len(set(selected)) != len(selected):
        raise ValueError('Invalid/duplicate dataset selection')
    threshold, validation_paths = None, []
    if calibration:
        cal = json.loads(Path(calibration).read_text())
        if cal['checkpoint_sha256'] != file_hash(checkpoint_path):
            raise ValueError('Calibration belongs to another checkpoint')
        threshold, validation_paths = float(cal['threshold']), cal['validation_paths']
        if not np.isfinite(threshold):
            raise ValueError('Invalid calibrated threshold')
    result = {'checkpoint_sha256': file_hash(checkpoint_path), 'suite': suite,
              'threshold': threshold, 'datasets': {}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mapping = {} if mapping is None else mapping
    for name in selected:
        folder = Path(root) / mapping.get(name, name)
        if (folder / 'test').is_dir():
            folder = folder / 'test'
        records = labelled_images(folder)
        paths = [p for p, _ in records]
        disjoint(state['train_paths'], validation_paths, paths)
        if suite == 'emerging' and not allow_other_count:
            counts = [sum(label == k for _, label in records) for k in (0, 1)]
            if counts != [1000, 1000]:
                raise ValueError('Table 5 expects 1000 real and 1000 fake images per generator')
        labels, scores = predict(model, TestImages(records, cfg), cfg)
        result['datasets'][name] = evaluate_scores(labels, scores, threshold)
        np.savez_compressed(output / (name + '.npz'), labels=labels, scores=scores, paths=np.asarray(list(map(str, paths))))
        print(name, json.dumps(result['datasets'][name]), flush=True)
    result['mean_ap'] = float(np.mean([v['ap'] for v in result['datasets'].values()]))
    result['mean_accuracy'] = None if threshold is None else float(np.mean([v['accuracy'] for v in result['datasets'].values()]))
    result['complete_suite'] = set(selected) == set(expected)
    write_json(output / 'metrics.json', result)
    return result


def import_legacy(source, cfg, train_manifest, output, device='cpu', cache_dir=None):
    """Adapter for encoder/MLP/decoder checkpoint dictionaries."""
    cfg.validate()
    old = torch.load(source, map_location='cpu', weights_only=True)
    required = {'encoder_state_dict', 'MLP_state_dict', 'decoder_state_dict'}
    if not required.issubset(old):
        raise ValueError('Expected CLIP-Flow encoder, projection and decoder state dictionaries')
    model = CLIPFlow(cfg, cache_dir=cache_dir).to(select_device(device))
    model.encoder.load_state_dict(old['encoder_state_dict'], strict=True)
    # A converted compact checkpoint relies on the official frozen CLIP weights.
    from models.clip import clip
    official, _ = clip.load('ViT-L/14', device='cpu', download_root=cache_dir)
    for key, value in official.state_dict().items():
        if not torch.equal(value, old['encoder_state_dict'][key].cpu()):
            raise ValueError('Encoder differs from frozen upstream CLIP; compact conversion would lose weights')
    model.projection.load_state_dict(old['MLP_state_dict'], strict=True)
    model.flow.load_state_dict(old['decoder_state_dict'], strict=True)
    paths = manifest(train_manifest)
    save_checkpoint(output, model, cfg, paths, epochs=0)
    return output
