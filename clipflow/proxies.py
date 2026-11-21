"""Spatial and spectral proxy-image transforms."""
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter


def band_mask(height, width, band, ratio, rng):
    if not 0 <= ratio <= 1:
        raise ValueError('ratio must be in [0, 1]')
    regions = {
        'low': (0, height // 4, 0, width // 4),
        'mid': (height // 4, 3 * height // 4, width // 4, 3 * width // 4),
        'high': (3 * height // 4, height, 3 * width // 4, width),
        'all': (0, height, 0, width),
    }
    y0, y1, x0, x1 = regions[band]
    area = (y1 - y0) * (x1 - x0)
    selected = rng.choice(area, int(np.ceil(area * ratio)), replace=False)
    mask = np.ones((height, width), dtype=np.float32)
    mask[y0 + selected // (x1 - x0), x0 + selected % (x1 - x0)] = 0
    return mask


def spectral(image, band, ratio, rng, phase_only=False, uint8_policy='wrap'):
    values = np.asarray(image, dtype=np.float32)
    spectrum = np.fft.fft2(values, axes=(0, 1))
    mask = band_mask(*values.shape[:2], band, ratio, rng)[..., None]
    if phase_only:
        # Zero selected phases while preserving spectral magnitudes.
        spectrum = np.abs(spectrum) * np.exp(1j * np.angle(spectrum) * mask)
    else:
        spectrum *= mask
    values = np.fft.ifft2(spectrum, axes=(0, 1)).real
    if uint8_policy == 'clip':
        values = np.clip(values, 0, 255)
    return Image.fromarray(values.astype(np.uint8))


def validation_proxy(image, inner=30, outer=100):
    """Held-out validation perturbation, distinct from training corner masks."""
    values = np.asarray(image, dtype=np.float32)
    h, w = values.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    mask = ~((radius >= inner) & (radius <= outer))
    spectrum = np.fft.fftshift(np.fft.fft2(values, axes=(0, 1)), axes=(0, 1))
    values = np.fft.ifft2(np.fft.ifftshift(spectrum * mask[..., None], axes=(0, 1)), axes=(0, 1)).real
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def make_proxy(image, cfg, rng=None):
    rng = np.random if rng is None else rng
    if cfg.proxy in ('frequency', 'phase'):
        return spectral(image, cfg.band, cfg.ratio, rng, cfg.proxy == 'phase', cfg.uint8_policy)
    if cfg.proxy == 'smooth':
        return image.filter(ImageFilter.GaussianBlur(cfg.blur_radius))
    if cfg.proxy == 'sharpen':
        return image.filter(ImageFilter.UnsharpMask(cfg.sharpen_radius, cfg.sharpen_percent))
    if cfg.proxy == 'noise':
        values = np.asarray(image, dtype=np.float32)
        values = values + rng.normal(0, cfg.noise_std, values.shape)
        return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
    if cfg.proxy == 'jitter':
        return ColorJitter(*cfg.jitter)(image)
    raise ValueError(cfg.proxy)
