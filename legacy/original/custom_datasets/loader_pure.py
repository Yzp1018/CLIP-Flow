import os
from random import random,choice,shuffle,randint
from scipy.ndimage.filters import gaussian_filter
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from io import BytesIO
from torchvision import transforms as T
from torch.utils.data import Dataset


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    #choice is the base function in random
    return choice(s)

def gaussian_blur(img, sigma):
    #用了scipy中的高斯模糊，kernel_size=sigma*6+1
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecoder(encimg, 1)
    return decimg[:, :, ::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2':cv2_jpg, 'pil':pil_jpg}

def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment(img, is_train):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    if random.random()<0.5:
        sig = sample_continuous([0.0, 3.0])
        if is_train == False:
            sig = 1.5
        gaussian_blur(img, sig)

    if random.random()<0.5:
        method = sample_discrete(['cv2', 'pil'])
        qual = sample_discrete(range(30, 101))
        if is_train == False:
            method = sample_discrete(['pil'])
            qual = 65
        img = jpeg_from_key(img, qual, method)
    return Image.fromarray(img)

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.x, self.y = self.load_dataset_folder()
        self.band = 'low'
        self.ratio = 1.0

        if is_train:
            self.tranform_x = T.Compose([
                T.Resize(c.img_size, Image.BILINEAR),
                T.Lambda(lambda img:data_augment(img, is_train)),
                T.RandomRotation(5),
                T.RandomCrop(c.crp_size),
                T.RandomHorizontalFlip()
            ])
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.BILINEAR),
                T.Lambda(lambda img:data_augment(img, is_train=False)),
                T.CenterCrop(c.crp_size)
            ])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')

        if not self.is_train:
            x = self.normalize(T.ToTensor()(self.transform_x(x)))
        else:
            x = self.transform_x(x)

            image_array = np.array(x).astype(np.complex64)
            freq_image = np.fft.fftn(image_array, axes=(0,1))
            height, width, _ = image_array.shape
            mask1 = self._create_balanced_mask(height, width)
            masked_freq_image = freq_image * mask1
            masked_image_array = np.fft.ifftn(masked_freq_image, axes=(0, 1)).real
            masked_image = Image.fromarray(masked_image_array.astype(np.uint8))

            x = self.normalize(T.ToTensor()(masked_image))
        return x,y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        if self.is_train:
            img_dir = ''
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)])
            img_fpath_list = img_fpath_list[::50]
            x.extend(img_fpath_list)
            y.extend([1] * len(img_fpath_list))
        else:
            img_dir = ''
            img_types = ['good', 'bad']
            for img_type in img_types:
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)])
                x.extend(img_fpath_list)

                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
        assert len(x) == len(y),'number of x and y should be same'
        return list(x),list(y)

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)
        if self.band == 'low':
            y_start, y_end = 0, height//4
            x_start, x_end = 0, width//4
        elif self.band == 'mid':
            y_start, y_end = height//4, 3 * height //4
            x_start, x_end = width //4, 3 * width //4
        elif self.band == 'high':
            y_start, y_end = 3 * height//4, height
            x_start, x_end = 3 * width //4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f'Invalid band:{self.band}')

        num_frequencies = int(np.ceil((y_end - y_start)*(x_end - x_start)*self.ratio))
        mask_frequencies_indices = np.random.permutation((y_end - y_start)*(x_end - x_start))[:num_frequencies]
        y_indices = mask_frequencies_indices // (x_end - x_start)
        x_indices = mask_frequencies_indices % (x_end - x_start)
        mask_single_channel = np.clip(np.random.normal(0.5, 1, (height, width)), 0, 1)
        mask1 = np.stack([mask_single_channel] * 3, axis=-1)
        mask[y_indices, x_indices, :] = mask1[y_indices, x_indices, :]

        return mask



