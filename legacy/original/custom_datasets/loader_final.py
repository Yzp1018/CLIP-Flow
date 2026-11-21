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
import cmath

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
                T.RandomCrop(c.crp_size),
                # T.RandomHorizontalFlip()
                T.ToTensor()
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
            # x = self.normalize(T.ToTensor()(self.transform_x(x)))
            x = self.normalize(self.transform_x(x))

        else:

            # x=self.transform_x(x)
            image_array = np.array(x).astype(np.complex64)
            image_float = np.complex64(image_array)
            # 获取图像的尺寸
            rows, cols, _ = image_float.shape
            # 获取图像的中心位置
            crow, ccol = rows // 2, cols // 2

            # 设置环形滤波的参数
            r1 = 0  # 内圆半径
            r2 = 20 # 外圆半径

            # 初始化输出图像
            output_image = np.zeros_like(image_float)
            mask_0 = np.ones((rows, cols), dtype=complex)
            y_start, y_end = 0, cols//4
            x_start, x_end = 0, rows//4
            num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * 1.0))
            mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
            y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
            x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
            # rand_nums = np.random.rand(num_frequencies)
            # rand_nums = np.exp(1j * rand_nums * cmath.pi)
            # mask_0[y_indices, x_indices] = rand_nums
            # mask_0[cols-1-y_indices, rows-1-x_indices] = rand_nums ** (-1)

            # 对每个通道进行FFT处理
            for i in range(3):  # 对RGB三个通道分别进行处理
                # 对每个通道应用FFT并将频域图像移到频域中心
                f = np.fft.fft2(image_float[:, :, i])
                f_ishift = f
                f_ishift[y_indices, x_indices] = f[y_indices, x_indices]/np.abs(f[y_indices, x_indices])
                # fshift = np.fft.fftshift(f)

                # # 创建环形掩码
                # # mask_0 = np.zeros((rows, cols),dtype=complex)
                # y0, x = np.ogrid[:rows, :cols]
                # area = (x - ccol) ** 2 + (y0 - crow) ** 2 >= r2 ** 2
                # mask_0[area] = 1
                # area1 = (x - ccol) ** 2 + (y0 - crow) ** 2 <= r1 ** 2
                # mask_0[area1] = 1
                # mask_0[~(area | area1)] = cmath.exp(1j * cmath.pi/4)

                # fshift = fshift * mask_0
                # 执行逆FFT
                # f_ishift = np.fft.ifftshift(fshift)
                img_back = np.real(np.fft.ifft2(f_ishift))
                # img_back = np.abs(img_back)  # 取绝对值，得到最终的图像

                # 将结果存储到输出图像
                output_image[:, :, i] = img_back

            real_image = np.array(np.real(output_image))
            output_image1 = np.uint8(np.clip(real_image, 0, 255))
            masked_image = Image.fromarray(output_image1)
            # x=self.normalize(T.ToTensor()(masked_image))
            # masked_image = masked_image.filter(ImageFilter.UnsharpMask(radius=5, percent=300))
            x = self.normalize(self.transform_x(masked_image))




        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

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

            img_dir = "/data/zpyuan/anomalib/datasets/MVTec/ProGAN/train/0_real"
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)])
            # # # if f.endswith('.png')])
            # # # img_fpath_list=['/home/liushuaichen/Workspace/zpyuan/anomalib/Workspace/zpyuan/anomalib/DRAEM/simplex_noise_256.png']
            img_fpath_list = img_fpath_list[::50]
            x.extend(img_fpath_list)
            print(len(img_fpath_list))
            #
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
