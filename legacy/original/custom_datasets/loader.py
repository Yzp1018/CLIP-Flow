import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from random import random, choice, shuffle
import cv2
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import torch.fft as fft

__all__ = ('MVTecDataset', 'StcDataset')


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper','ProGAN_mini','FACE','ProGAN_10k',
                     'ProGAN']

STC_CLASS_NAMES = ['01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12'] #, '13' - no ground-truth]


class StcDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in STC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, STC_CLASS_NAMES)
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        #
        if is_train:
            self.dataset_path = os.path.join(c.data_path, 'training')
            self.dataset_vid = os.path.join(self.dataset_path, 'videos')
            self.dataset_dir = os.path.join(self.dataset_path, 'frames')
            self.dataset_files = sorted([f for f in os.listdir(self.dataset_vid) if f.startswith(self.class_name)])
            if not os.path.isdir(self.dataset_dir):
                os.mkdir(self.dataset_dir)
            done_file = os.path.join(self.dataset_path, 'frames_{}.pt'.format(self.class_name))
            print(done_file)
            H, W = 480, 856
            if os.path.isfile(done_file):
                assert torch.load(done_file) == len(self.dataset_files), 'train frames are not processed!'
            else:
                count = 0
                for dataset_file in self.dataset_files:
                    print(dataset_file)
                    data = read_video(os.path.join(self.dataset_vid, dataset_file)) # read video file entirely -> mem issue!!!
                    vid = data[0] # weird read_video that returns byte tensor in format [T,H,W,C]
                    fps = data[2]['video_fps']
                    print('video mu/std: {}/{} {}'.format(torch.mean(vid/255.0, (0,1,2)), torch.std(vid/255.0, (0,1,2)), vid.shape))
                    assert [H, W] == [vid.size(1), vid.size(2)], 'same H/W'
                    dataset_file_dir = os.path.join(self.dataset_dir, os.path.splitext(dataset_file)[0])
                    os.mkdir(dataset_file_dir)
                    count = count + 1
                    for i, frame in enumerate(vid):
                        filename = '{0:08d}.jpg'.format(i)
                        write_jpeg(frame.permute((2, 0, 1)), os.path.join(dataset_file_dir, filename), 80)
                torch.save(torch.tensor(count), done_file)
            #
            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.dataset_path = os.path.join(c.data_path, 'testing')
            self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.ToPILImage(),
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.normalize(self.transform_x(x))
        if y == 0: #self.is_train:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = self.transform_mask(mask)
        #
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = list(), list(), list()
        img_dir = os.path.join(self.dataset_path, 'frames')
        img_types = sorted([f for f in os.listdir(img_dir) if f.startswith(self.class_name)])
        gt_frame_dir = os.path.join(self.dataset_path, 'test_frame_mask')
        gt_pixel_dir = os.path.join(self.dataset_path, 'test_pixel_mask')
        for i, img_type in enumerate(img_types):
            print('Folder:', img_type)
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg')])
            x.extend(img_fpath_list)
            # labels for every test image
            if phase == 'test':
                gt_pixel = np.load('{}.npy'.format(os.path.join(gt_pixel_dir, img_type)))
                gt_frame = np.load('{}.npy'.format(os.path.join(gt_frame_dir, img_type)))
                if i == 0:
                    m = gt_pixel
                    y = gt_frame
                else:
                    m = np.concatenate((m, gt_pixel), axis=0)
                    y = np.concatenate((y, gt_frame), axis=0)
                #
                mask = [e for e in m] # np.expand_dims(e, axis=0)
                assert len(x) == len(y), 'number of x and y should be same'
                assert len(x) == len(mask), 'number of x and mask should be same'
            else:
                mask.extend([None] * len(img_fpath_list))
                y.extend([0] * len(img_fpath_list))
        #
        return list(x), list(y), list(mask)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment(img,c):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < 0.5:
        sig = sample_continuous([0.0,3.0])
        if c.action_type == 'norm_test':
            sig = 1.5
        gaussian_blur(img, sig)

    if random() < 0.5:
        method = sample_discrete(['cv2','pil'])
        qual = sample_discrete(range(50,101))
        if c.action_type == 'norm_test':
            qual = 85
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.BICUBIC),
                T.Lambda(lambda img: data_augment(img,c)),
                # T.RandomRotation(5),
                T.RandomCrop(c.crp_size),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.BICUBIC),
                #T.Lambda(lambda img: data_augment(img,c)),
                T.CenterCrop(c.crp_size),
                T.ToTensor()
            ])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()
        ])
        self.totensor = T.ToTensor()

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x).convert("RGB")
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        # x = self.normalize(self.transform_x(x))
        # image = x
        # x = self.transform_x(x)

        x = T.ToTensor()(x)
        f_transform = fft.fftn(x, dim=(-2, -1))
        f_shift = fft.fftshift(f_transform)

        # 获取图像大小
        rows, cols = x.shape[-2:]
        crow, ccol = rows // 2, cols // 2

        # 创建一个掩码，中心部分为1，其余部分为0
        mask = torch.zeros((rows, cols))
        # r = int(filter_strength * min(rows, cols) / 1000000)
        r1 = 54.99
        r2 = 55
        center = [crow, ccol]
        p = torch.arange(rows, dtype=torch.float32)
        q = torch.arange(cols, dtype=torch.float32)
        xx, yy = torch.meshgrid(p, q, indexing='ij')
        mask_area = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 >= r2 ** 2
        mask[mask_area] = 1
        mask_area2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= r1 ** 2
        mask[mask_area2] = 1

        # 将掩码应用于频谱
        f_shift_filtered = f_shift * mask.to(f_shift.device)

        # 逆变换回时域
        f_ishift = fft.ifftshift(f_shift_filtered)
        # image = fft.ifftn(f_ishift, dim=(-2, -1)).real
        # image = T.ToPILImage()(image)
        image = fft.ifftn(f_ishift, dim=(-2, -1))

        # 使用绝对值恢复强度信息
        image = T.ToPILImage()(torch.abs(image))
        image = self.transform_x(image)
        image = self.normalize(image)
        #

        # image = self.totensor(image)
        # image = (image-0.5)/1.0
        # x = np.array(x).astype(np.float32)
        # x = x / 255.0
        # image = np.zeros_like(x)
        # for i in range(3):
        #     im = x[:, :, i]
        #     fft_img = np.fft.fft2(im)
        #     fft_img = np.log(np.abs(fft_img) + 1e-3)
        #     fft_min = np.percentile(fft_img, 5)
        #     fft_max = np.percentile(fft_img, 95)
        #     fft_img = (fft_img - fft_min) / (fft_max - fft_min + 1e-6)
        #     fft_img = (fft_img - 0.5) * 2
        #     fft_img[fft_img < -1] = -1
        #     fft_img[fft_img > 1] = 1
        #
        #     fft_img = np.fft.fftshift(fft_img)
        #     fft_img[:21, :] = 0
        #     fft_img[:, :21] = 0
        #     fft_img[203:, :] = 0
        #     fft_img[:, 203:] = 0
        #     fft_img[57:177, 57:177] = 0
        #     fft_img = np.fft.fftshift(fft_img)
        #     image[:, :, i] = fft_img
        # image = np.transpose(image, (2, 0, 1))
        #
        # image = self.totensor(image)
        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return image, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
