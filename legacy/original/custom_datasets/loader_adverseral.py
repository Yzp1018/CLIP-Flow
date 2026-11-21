import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T

__all__ = ('MVTecDataset', 'StcDataset')

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        # assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.ratio = 1.0
        self.band = 'mid'
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        # x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        # x = self.normalize(self.transform_x(x))
        #
        image_array = np.array(x).astype(np.complex64)
        freq_image = np.fft.fftn(image_array, axes=(0, 1))

        height, width, _ = image_array.shape

        mask = self._create_balanced_mask(height, width)
        self.masked_freq_image = freq_image * mask
        masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        x = self.normalize(self.transform_x(masked_image))

        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)

        # Determine the region of the frequency domain to mask
        if self.band == 'low':
            y_start, y_end = 0, height // 4
            x_start, x_end = 0, width // 4
        elif self.band == 'mid':
            y_start, y_end = 3 * height // 7, 4 * height // 7
            x_start, x_end = 3 * width // 7, 4 * width // 7
        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f"Invalid band: {self.band}")

        num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
        mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
        y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
        x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

        mask[y_indices, x_indices, :] = 0
        return mask

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        if(self.is_train):
            img_dir = "/data/zpyuan/anomalib/create_adverseral_sample/checkpoints/visualization"

            img_fpath_list = sorted([os.path.join(img_dir, f)
                                         for f in os.listdir(img_dir)])
                # if f.endswith('.png')])
            x.extend(img_fpath_list)

            y.extend([1] * len(img_fpath_list))
            mask.extend([1] * len(img_fpath_list))
            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/train/good"
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)])
            # if f.endswith('.png')])
            x.extend(img_fpath_list)

            y.extend([0] * len(img_fpath_list))
            mask.extend([0] * len(img_fpath_list))

        else:
            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/test"
            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)])
                # if f.endswith('.png')])
                x.extend(img_fpath_list)

                # load gt labels
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([0] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    mask.extend([0] * len(img_fpath_list))


        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
