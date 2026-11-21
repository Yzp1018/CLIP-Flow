import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from random import random, choice, shuffle,randint
import random
import cv2
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
from PIL import Image, ImageFilter

__all__ = ('MVTecDataset', 'StcDataset')

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'ProGAN_mini', 'FACE', 'ProGAN_10k',
                     'ProGAN']

STC_CLASS_NAMES = ['01', '02', '03', '04', '05', '06',
                   '07', '08', '09', '10', '11', '12']  # , '13' - no ground-truth]


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
                    data = read_video(
                        os.path.join(self.dataset_vid, dataset_file))  # read video file entirely -> mem issue!!!
                    vid = data[0]  # weird read_video that returns byte tensor in format [T,H,W,C]
                    fps = data[2]['video_fps']
                    print('video mu/std: {}/{} {}'.format(torch.mean(vid / 255.0, (0, 1, 2)),
                                                          torch.std(vid / 255.0, (0, 1, 2)), vid.shape))
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
        if y == 0:  # self.is_train:
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
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg')])
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
                mask = [e for e in m]  # np.expand_dims(e, axis=0)
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
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


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


def data_augment(img, is_train):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    # if random() < 0.5:
    if random.random()<0.1:
        sig = sample_continuous([0.0, 1.0])
        # sig=3.0
        if is_train == False:
            sig = 0.5
        gaussian_blur(img, sig)

    # img = Image.fromarray(img).filter(ImageFilter.UnsharpMask(radius=5, percent=300))
    # img=np.array(img)

    # if random() < 0.5:
    if random.random()<0.1:
        method = sample_discrete(['cv2', 'pil'])
        # qual = sample_discrete(range(50, 101))
        qual = sample_discrete(range(30,101))
        # qual=80
        if is_train == False:
            method = sample_discrete(['pil'])
            qual = 95
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        # assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name,
        #                                                                                    MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        self.band='low'
        self.ratio=1.0
        # self.image_dir='/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/ProGAN_generator/Images_Gen'
        # self.noise_img=sorted([os.path.join(self.image_dir, f)
        #                              for f in os.listdir(self.image_dir)])
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.BILINEAR),
                # T.Lambda(lambda img: data_augment(img, is_train)),
                # T.RandomRotation(5),
                T.RandomCrop(c.crp_size)
                # T.RandomHorizontalFlip()
                # T.ToTensor()
            ])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.BILINEAR),
                # T.Lambda(lambda img: data_augment(img, is_train)),
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
        # x = Image.open(x).convert('RGB')
        try:
            x = Image.open(x).convert("RGB")
        except:
            x = '/home/liushuaichen/Workspace/zpyuan/anomalib/datasets/MVTec/CNNdetection/imle/test/good/00100001.png'
            x = Image.open(x).convert("RGB")
            self.y[idx]=0
            y=0

        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        if not self.is_train:
            # x=x.filter(ImageFilter.UnsharpMask(radius=7, percent=100))

            x = self.normalize(self.transform_x(x))
        elif y==0:
            x = self.normalize(T.ToTensor()(self.transform_x(x)))
            # x = self.normalize(self.transform_x(x))
        else:
            # x = self.normalize(T.ToTensor()(self.transform_x(x)))
            # if y==0:
            #     self.ratio=0.7
            # if y==1:
            #     self.ratio=1.0
            if random.random()<0:
                # x=self.transform_x(x)
                image_array = np.array(x).astype(np.complex64)
                freq_image = np.fft.fftn(image_array, axes=(0, 1))

                height, width, _ = image_array.shape

                mask1 = self._create_balanced_mask(height, width,y)
                self.masked_freq_image = freq_image * mask1
                masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real
                masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
                # x=self.normalize(T.ToTensor()(masked_image))
                # masked_image = masked_image.filter(ImageFilter.UnsharpMask(radius=5, percent=300))
                x = self.normalize(self.transform_x(masked_image))
            else:
                #先模糊再锐化，先降分辨率再超分辨率，针对的是san类数据集，试一下是否可以有一定的提升
                x=self.transform_x(x)
                # x=np.array(x).astype(np.uint8)
                # gaussian_blur(x, 0.45)
                # x=Image.fromarray(x)
                # x =x.filter(ImageFilter.UnsharpMask(radius=3, percent=100))
                x = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(x)
                x = T.ToTensor()(x)
                x = self.normalize(x)
                # random_path = random.choice(self.noise_img)
                # noisy=Image.open(random_path).convert("RGB")
                # noisy= np.array(noisy).astype(np.float32)
                # x = np.add(0.8 * noisy , 0.2 * image)
                # x = Image.fromarray(x.astype(np.uint8))
            # x = self.normalize(self.transform_x(x))


        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        # img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        # gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        #
        # img_types = sorted(os.listdir(img_dir))
        # for img_type in img_types:
        #
        #     # load images
        #     img_type_dir = os.path.join(img_dir, img_type)
        #     if not os.path.isdir(img_type_dir):
        #         continue
        #     img_fpath_list = sorted([os.path.join(img_type_dir, f)
        #                              for f in os.listdir(img_type_dir)
        #                              if f.endswith('.png')])
        #     x.extend(img_fpath_list)
        #
        #     # load gt labels
        #     if img_type == 'good':
        #         y.extend([0] * len(img_fpath_list))
        #         mask.extend([None] * len(img_fpath_list))
        #     else:
        #         y.extend([1] * len(img_fpath_list))
        #         gt_type_dir = os.path.join(gt_dir, img_type)
        #         img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
        #         gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
        #                          for img_fname in img_fname_list]
        #         mask.extend(gt_fpath_list)
        if (self.is_train):
            # img_dir = '/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/DRAEM/bluegray'
            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/train/0_real"

            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)])
            # # if f.endswith('.png')])
            img_fpath_list=img_fpath_list[::50]
            x.extend(img_fpath_list)
            print(len(img_fpath_list))

            y.extend([1] * len(img_fpath_list))
            mask.extend([1] * len(img_fpath_list))
            # img_dir='/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/ProGAN_generator/Images_Gen'
            # img_dir = "/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/colortrans"
            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/train/0_real"
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)])
            # # if f.endswith('.png')])
            # # img_fpath_list=['/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/DRAEM/simplex_noise_256.png']
            img_fpath_list = img_fpath_list[::50]
            x.extend(img_fpath_list)
            print(len(img_fpath_list))
            # #
            # # choices = np.arange(0.1, 1.1, 0.1)  # 创建 [0, 0.1, 0.2, ..., 1] 的范围
            # # probabilities = [0.0020, 0.0020, 0.0025, 0.0035, 0.005, 0.005, 0.01, 0.01, 0.08, 0.08, 0.8]
            # # random_array = np.random.choice(choices, len(img_fpath_list))
            # # y.extend(random_array)
            y.extend([0] * len(img_fpath_list))
            mask.extend([0] * len(img_fpath_list))

            # indices = list(range(len(x)))
            # random.shuffle(indices)
            # x = [x[i] for i in indices]
            # y = [y[i] for i in indices]
        else:
            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/test"
            # img_dir='/data/zpyuan/anomalib/datasets/MVTec/CNNdetection/ldm_200/test'

            # img_dir  = os.path.join("/home/liushuaichen/Workspace/zpyuan/anomalib/datasets/MVTec/CNNdetection", self.class_name, 'test')
            # img_types = sorted(os.listdir(img_dir))
            img_types=['good','bad']
            # img_types=['ImageNet']
            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)])
                # if f.endswith('.png')])
                img_fpath_list = img_fpath_list[:]
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


    # def _create_balanced_mask(self, height, width,y):
    #     mask = np.ones((height, width, 3), dtype=np.complex64)
    #     # if y == 0:
    #     #     self.ratio = 1.0
    #     # if y == 1:
    #     #     self.ratio = 0.7
    #     ban=random.random()
    #     if ban<0.95:
    #         self.band='low'
    #         # self.ratio=1.0
    #     # elif ban<0.94:
    #     #     self.band='high'
    #     #     self.ratio = 1.0
    #     # elif ban<0.96:
    #     #     self.band='mid'
    #     #     self.ratio=0.8
    #     else:
    #         self.band='high'
    #         # self.ratio=1.0
    #         # self.ratio=0.25
    #     # Determine the region of the frequency domain to mask
    #     if self.band == 'low':
    #         y_start, y_end = 0, height // 4
    #         x_start, x_end = 0, width // 4
    #     elif self.band == 'mid':
    #         y_start, y_end = 1* height // 4, 3 * height // 4
    #         x_start, x_end = 1 * width // 4, 3 * width // 4
    #     elif self.band == 'high':
    #         y_start, y_end = 3 * height // 4, height
    #         x_start, x_end = 3 * width // 4, width
    #     elif self.band == 'all':
    #         y_start, y_end = 0, height
    #         x_start, x_end = 0, width
    #     else:
    #         raise ValueError(f"Invalid band: {self.band}")
    #     if random.random()<0.2:
    #         num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
    #         mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
    #         y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
    #         x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
    #         # mask[y_indices, x_indices, :] = 0.5
    #         mask_single_channel = np.clip(np.random.normal(0.5, 1, (height, width)), 0, 1)
    #         mask1 = np.stack([mask_single_channel] * 3, axis=-1)
    #         mask[y_indices, x_indices, :] = mask1[y_indices, x_indices, :]
    #     else:
    #         num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
    #         mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
    #         y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
    #         x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
    #         # mask[y_indices, x_indices, :] = 0.5
    #         mask_single_channel = np.clip(np.random.normal(7, 10, (height, width)), 5, 10)
    #         mask1 = np.stack([mask_single_channel] * 3, axis=-1)
    #         mask[y_indices, x_indices, :] = mask1[y_indices, x_indices, :]
    #     # num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
    #     # mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
    #     # y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
    #     # x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
    #     # mask[y_indices, x_indices, :] = 0
    #     # mask[y_start:y_end, x_start:y_end, :] = 5
    #
    #     # mask[y_indices, x_indices, :] = 0
    #     return mask

    def _create_balanced_mask(self, height, width,y):
        mask = np.ones((height, width, 3), dtype=np.complex64)
        if self.band == 'low':
            y_start, y_end = 0, height // 4
            x_start, x_end = 0, width // 4
        elif self.band == 'mid':
            y_start, y_end = 1* height // 4, 3 * height // 4
            x_start, x_end = 1 * width // 4, 3 * width // 4
        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f"Invalid band: {self.band}")
        if random.random()<1:
            num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
            mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
            y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
            x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
            mask[y_indices, x_indices, :] = 0
        else:
            beishu=randint(3,10)
            mask[y_start:y_end, x_start:y_end, :] = beishu
        # num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
        # mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
        # y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
        # x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
        # mask[y_indices, x_indices, :] = 0
        # mask[y_start:y_end, x_start:y_end, :] = 5

        # mask[y_indices, x_indices, :] = 0
        return mask
