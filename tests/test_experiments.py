import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from clipflow.data import TrainingImages, labelled_images, manifest
from clipflow.engine import train, calibrate, evaluate, load_checkpoint, file_hash
from clipflow.experiments import GROUPS, plan
from clipflow.metrics import calibrate_threshold, evaluate_scores
from clipflow.network import CLIPFlow, likelihood_loss
from clipflow.proxies import band_mask, make_proxy, validation_proxy
from clipflow.settings import Settings


class TinyEncoder(nn.Module):
    """Synthetic feature extractor; never substitutes for CLIP in real runs."""
    def __init__(self):
        super().__init__()
        self.register_buffer('basis', torch.linspace(0.1, 1, 768).reshape(3, 256))

    def encode_image(self, image):
        channels = image.mean((2, 3))
        return (channels[..., None] * self.basis).reshape(-1, 768)


def make_images(folder, count, seed):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(count):
        path = folder / (str(i) + '.png')
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        paths.append(str(path.resolve()))
    return paths


def save_manifest(path, paths):
    path.write_text(json.dumps(paths))
    return path


@pytest.mark.parametrize('mode,expected', [('N', [-1., -1., 0., 0.]), ('P', [0., 0., 1., 1.]), ('NP', [-1., -1., 1., 1.])])
def test_objective_gradient_directions(mode, expected):
    scores = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    likelihood_loss(scores, labels, mode).backward()
    assert torch.equal(scores.grad.sign(), torch.tensor(expected))


def test_threshold_ties_match_exhaustive_balanced_search():
    rng = np.random.default_rng(13)
    for _ in range(30):
        labels = np.r_[np.zeros(9, int), np.ones(5, int)]
        scores = rng.integers(-3, 4, len(labels)).astype(float)
        threshold, quality = calibrate_threshold(labels, scores)
        possible = np.r_[scores.min(), np.nextafter(np.unique(scores), np.inf)]
        def balanced(t):
            pred = scores >= t
            return ((pred[labels == 1] == 1).mean() + (pred[labels == 0] == 0).mean()) / 2
        assert quality == pytest.approx(max(map(balanced, possible)))
        assert balanced(threshold) == pytest.approx(quality)


@pytest.mark.parametrize('dimension,count', [(768,1478400),(512,788992),(256,296192),(128,123520),
                                          (64,55616),(32,26272),(16,12752),(8,6280),(4,3116),(2,1552)])
def test_parameter_counts_match_paper_figure4(dimension, count):
    model = CLIPFlow(Settings(dimension=dimension), encoder=TinyEncoder())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == count


def test_no_dr_parameter_count_records_unresolved_figure_discrepancy():
    model = CLIPFlow(Settings(dimension=0), encoder=TinyEncoder())
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 887808
    assert 888576 - count == 768  # Figure 4 labels 888,576; no fictitious parameters added.


def test_frozen_encoder_and_consistent_training_inference():
    model = CLIPFlow(Settings(dimension=8), encoder=TinyEncoder())
    image = torch.rand(3, 3, 16, 16)
    model.train()
    before = model(image)
    before.mean().backward()
    assert not model.encoder.training
    assert all(p.grad is None for p in model.encoder.parameters())
    model.eval()
    assert torch.allclose(before, model(image))


@pytest.mark.parametrize('band', ['low', 'mid', 'high', 'all'])
def test_frequency_mask_exact_ratio_and_region(band):
    mask = band_mask(32, 32, band, 0.1, np.random.default_rng(1))
    y0,y1,x0,x1 = {'low':(0,8,0,8),'mid':(8,24,8,24),'high':(24,32,24,32),'all':(0,32,0,32)}[band]
    assert (mask == 0).sum() == int(np.ceil((y1-y0)*(x1-x0)*0.1))
    outside = mask.copy()
    outside[y0:y1,x0:x1] = 1
    assert np.all(outside == 1)


@pytest.mark.parametrize('proxy', ['frequency', 'phase', 'smooth', 'sharpen', 'noise', 'jitter'])
def test_proxies_preserve_valid_rgb_shape(proxy):
    image = Image.fromarray(np.random.default_rng(7).integers(0,256,(32,32,3),dtype=np.uint8))
    result = make_proxy(image, Settings(proxy=proxy), np.random.default_rng(10))
    assert result.mode == 'RGB' and result.size == image.size
    assert np.asarray(result).dtype == np.uint8


def test_zero_frequency_mask_is_identity():
    image = Image.fromarray(np.random.default_rng(4).integers(0,256,(32,32,3),dtype=np.uint8))
    result = np.asarray(make_proxy(image, Settings(ratio=0), np.random.default_rng(9)))
    # FFT roundoff followed by integer conversion is at most one intensity level.
    assert np.abs(result.astype(float)-np.asarray(image)).max() <= 1


def test_pure_phase_keeps_spectral_magnitude_before_quantization(monkeypatch):
    import clipflow.proxies as proxies
    captured = {}
    original = np.fft.ifft2
    def capture(spectrum, **kw):
        captured['spectrum'] = spectrum.copy()
        return original(spectrum, **kw)
    monkeypatch.setattr(proxies.np.fft, 'ifft2', capture)
    values = np.random.default_rng(2).integers(0,256,(32,32,3),dtype=np.uint8)
    make_proxy(Image.fromarray(values), Settings(proxy='phase',ratio=1), np.random.default_rng(3))
    expected = np.abs(np.fft.fft2(values.astype(np.float32), axes=(0,1)))
    assert np.allclose(np.abs(captured['spectrum']), expected)


def test_train_calibrate_evaluate_pipeline_and_overlap_guards(tmp_path):
    torch.set_num_threads(1)
    cfg = Settings(mode='NP', epochs=1, dimension=8, resize=32, crop=24, batch_size=4)
    training = make_images(tmp_path/'train', 4, 1)
    validation = make_images(tmp_path/'val', 3, 2)
    train_manifest = save_manifest(tmp_path/'train.json', training)
    val_manifest = save_manifest(tmp_path/'val.json', validation)
    run = tmp_path/'run'
    _, history = train(cfg, train_manifest, run, encoder=TinyEncoder(), allow_other_count=True)
    assert len(history) == 1
    checkpoint = run/'last.pt'
    cal_path = run/'calibration.json'
    cal = calibrate(checkpoint, val_manifest, cal_path, encoder=TinyEncoder())
    assert cal['checkpoint_sha256'] == file_hash(checkpoint)
    make_images(tmp_path/'test/progan/test/0_real', 2, 3)
    make_images(tmp_path/'test/progan/test/1_fake', 2, 4)
    result = evaluate(checkpoint, tmp_path/'test', run/'evaluation', cal_path,
                      datasets=['progan'], encoder=TinyEncoder())
    assert result['threshold'] == cal['threshold']
    assert result['datasets']['progan']['count'] == 4
    assert 0 <= result['mean_ap'] <= 1 and result['mean_accuracy'] is not None
    assert not result['complete_suite']
    with pytest.raises(ValueError, match='overlap'):
        calibrate(checkpoint, train_manifest, cal_path, encoder=TinyEncoder())
    wrong = dict(cal, checkpoint_sha256='wrong')
    bad = tmp_path/'bad.json'; bad.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='another checkpoint'):
        evaluate(checkpoint, tmp_path/'test', run/'bad', bad, datasets=['progan'], encoder=TinyEncoder())
    model, _, state = load_checkpoint(checkpoint, encoder=TinyEncoder())
    assert state['completed_epochs'] == 1
    assert torch.isfinite(model.score(torch.rand(2,3,24,24))).all()


def test_nested_dataset_labels_and_missing_class(tmp_path):
    make_images(tmp_path/'progan/class1/0_real', 1, 10)
    make_images(tmp_path/'progan/class1/1_fake', 1, 11)
    assert [y for _,y in labelled_images(tmp_path/'progan')] == [0,1]
    make_images(tmp_path/'incomplete/0_real', 1, 12)
    with pytest.raises(ValueError, match='Both real and fake'):
        labelled_images(tmp_path/'incomplete')


def test_all_modes_have_expected_natural_proxy_counts(tmp_path):
    paths = make_images(tmp_path/'real', 3, 9)
    for mode, count, labels in [('N',3,[0]*3),('P',3,[1]*3),('NP',6,[0]*3+[1]*3)]:
        cfg = Settings(mode=mode, resize=32, crop=24)
        dataset = TrainingImages(paths,cfg)
        assert len(dataset)==count
        assert [dataset[i][1] for i in range(len(dataset))] == labels


def test_all_experiment_configs_and_same_p_model_for_emerging():
    root = Path(__file__).resolve().parents[1]
    for name in set(n for names in GROUPS.values() for n in names):
        Settings(**json.loads((root/'configs'/(name+'.json')).read_text())).validate()
    commands = plan(list(GROUPS), 'train.json','validation.json','test','runs','emerging')
    training = [c for c in commands if c[3]=='train']
    assert len(training)==len(set(c[c.index('--config')+1] for c in training))
    last=commands[-1]
    assert last[last.index('--checkpoint')+1]=='runs/P/last.pt'
    assert '--suite' in last and 'emerging' in last


def test_vendored_clip_import_without_downloading_weights():
    from models.clip import clip
    assert 'ViT-L/14' in clip.available_models()


def test_actual_clip_component_with_random_small_architecture():
    from models.clip.model import CLIP
    encoder = CLIP(embed_dim=768, image_resolution=32, vision_layers=1,
                   vision_width=64, vision_patch_size=16, context_length=8,
                   vocab_size=100, transformer_width=64, transformer_heads=1, transformer_layers=1)
    model = CLIPFlow(Settings(dimension=8), encoder=encoder)
    result = model.score(torch.rand(2,3,32,32))
    assert result.shape == (2,) and torch.isfinite(result).all()
