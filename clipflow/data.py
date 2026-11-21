import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .proxies import make_proxy, validation_proxy

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def images(root):
    paths = sorted(p.resolve() for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No images found: ' + str(root))
    return paths


def manifest(path):
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise ValueError('Manifest must be a nonempty JSON list of image paths')
    paths = [(path.parent / v).resolve() for v in values]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate images in manifest')
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def fingerprint(paths):
    # Hash resolved membership paths, rather than image contents.
    return hashlib.sha256('\n'.join(sorted(map(str, paths))).encode()).hexdigest()


def disjoint(*groups):
    seen = set()
    for group in groups:
        current = set(map(lambda p: str(Path(p).resolve()), group))
        overlap = seen & current
        if overlap:
            raise ValueError('Data split overlap: ' + sorted(overlap)[0])
        seen.update(current)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def image_tensor(image, cfg, training=False):
    crop = T.RandomCrop(cfg.crop) if training else T.CenterCrop(cfg.crop)
    return T.Compose([crop, T.ToTensor(), T.Normalize(MEAN, STD)])(image)


def open_resized(path, cfg):
    with Image.open(path) as source:
        image = source.convert('RGB')
    return T.Resize((cfg.resize, cfg.resize), interpolation=InterpolationMode.BILINEAR)(image)


class TrainingImages(Dataset):
    def __init__(self, paths, cfg):
        self.paths, self.cfg = paths, cfg

    def __len__(self):
        return len(self.paths) * (2 if self.cfg.mode == 'NP' else 1)

    def __getitem__(self, index):
        count = len(self.paths)
        proxy = self.cfg.mode == 'P' or (self.cfg.mode == 'NP' and index >= count)
        image = open_resized(self.paths[index % count], self.cfg)
        if proxy:
            image = make_proxy(image, self.cfg)
        return image_tensor(image, self.cfg, True), int(proxy)


class ValidationImages(Dataset):
    def __init__(self, paths, cfg):
        self.paths, self.cfg = paths, cfg

    def __len__(self):
        return 2 * len(self.paths)

    def __getitem__(self, index):
        count = len(self.paths)
        proxy = index >= count
        image = open_resized(self.paths[index % count], self.cfg)
        if proxy:
            image = validation_proxy(image)
        return image_tensor(image, self.cfg), int(proxy)


def labelled_images(root):
    """Support flat or class-nested good/bad and 0_real/1_fake layouts."""
    labels = {'good': 0, '0_real': 0, 'real': 0, 'bad': 1, '1_fake': 1, 'fake': 1}
    root = Path(root).resolve()
    records = []
    for p in images(root):
        candidates = [labels[part.lower()] for part in p.relative_to(root).parts[:-1] if part.lower() in labels]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one real/fake label directory for ' + str(p))
        records.append((p, candidates[0]))
    if set(label for _, label in records) != {0, 1}:
        raise ValueError('Both real and fake test images are required: ' + str(root))
    return records


class TestImages(Dataset):
    def __init__(self, records, cfg):
        self.records, self.cfg = records, cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        return image_tensor(open_resized(path, self.cfg), self.cfg), label
