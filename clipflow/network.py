import math

import torch
from torch import nn
from torch.nn import functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def subnet(din, dout):
    return nn.Sequential(nn.Linear(din, 2 * din), nn.ReLU(), nn.Linear(2 * din, dout))


class CLIPFlow(nn.Module):
    def __init__(self, cfg, encoder=None, cache_dir=None):
        super().__init__()
        cfg.validate()
        if encoder is None:
            from models.clip import clip
            encoder, _ = clip.load('ViT-L/14', device='cpu', download_root=cache_dir)
        self.encoder = encoder.eval()
        self.encoder.requires_grad_(False)
        self.projection = nn.Linear(768, cfg.dimension) if cfg.dimension else nn.Identity()
        if cfg.dimension:
            nn.init.orthogonal_(self.projection.weight)
        self.normalize = cfg.normalize
        self.flow = Ff.SequenceINN(cfg.dimension or 768)
        for _ in range(cfg.coupling_blocks):
            self.flow.append(Fm.AllInOneBlock, subnet_constructor=subnet,
                             affine_clamping=cfg.clamp, global_affine_type='SOFTPLUS', permute_soft=False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder.encode_image(images)
        features = self.projection(features.float())
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        latent, logdet = self.flow(features)
        dimension = features.shape[1]
        return (-0.5 * latent.square().sum(1) + logdet - 0.5 * dimension * math.log(2 * math.pi)) / dimension

    def score(self, images):
        return -self(images)


def likelihood_loss(logp, labels, mode):
    terms = []
    if mode in ('N', 'NP') and (labels == 0).any():
        terms.append(-logp[labels == 0].mean())
    if mode in ('P', 'NP') and (labels == 1).any():
        terms.append(logp[labels == 1].mean())
    if not terms:
        raise ValueError('No samples for selected likelihood objective')
    return sum(terms)
