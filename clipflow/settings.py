from dataclasses import asdict, dataclass


BENCHMARKS = (
    'progan', 'cyclegan', 'biggan', 'stylegan', 'gaugan', 'stargan',
    'deepfake', 'seeingdark', 'san', 'crn', 'imle', 'guided',
    'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27',
    'glide_50_27', 'glide_100_10', 'dalle',
)
EMERGING = ('dalle3', 'deepfloyd', 'realvisxl_v4', 'sd15')


@dataclass
class Settings:
    mode: str = 'NP'
    proxy: str = 'frequency'
    band: str = 'low'
    ratio: float = 0.1
    dimension: int = 128  # 0 means no dimension-reduction layer
    normalize: bool = True
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-4
    coupling_blocks: int = 1
    clamp: float = 1.9
    resize: int = 256
    crop: int = 224
    seed: int = 1000
    workers: int = 0
    noise_std: float = 5.0
    blur_radius: float = 1.0
    sharpen_radius: float = 5.0
    sharpen_percent: int = 300
    jitter: tuple = (0.5, 0.5, 0.5, 0.1)
    uint8_policy: str = 'wrap'

    def validate(self):
        if self.mode not in ('N', 'P', 'NP'):
            raise ValueError('mode must be N, P or NP')
        if self.proxy not in ('frequency', 'phase', 'smooth', 'sharpen', 'noise', 'jitter'):
            raise ValueError('unsupported proxy')
        if self.band not in ('low', 'mid', 'high', 'all') or not 0 <= self.ratio <= 1:
            raise ValueError('invalid frequency band/ratio')
        if self.uint8_policy not in ('wrap', 'clip'):
            raise ValueError('uint8_policy must be wrap or clip')
        if self.dimension < 0 or (self.dimension and self.dimension % 2) or self.dimension > 768:
            raise ValueError('dimension must be 0 or an even number <= 768')
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.coupling_blocks < 1:
            raise ValueError('invalid training settings')
        if not 0 < self.crop <= self.resize or self.workers < 0:
            raise ValueError('invalid image size or worker count')
        if self.mode == 'NP' and self.batch_size < 2:
            raise ValueError('NP requires batch_size >= 2')
        return self

    def to_dict(self):
        return asdict(self)
