import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, accuracy_score
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *
from models.clip import clip
from copy import deepcopy

OUT_DIR = './viz/'

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres

def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, N):
    P = c.condition_vec
    decoders = decoders.train()
    # adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=False)):
            # warm-up learning rate c.hide_tqdm_bar
            # lr = warmup_learning_rate(c, epoch, i + sub_epoch * I, I * c.sub_epochs, optimizer)
            lr = c.lr
            image = image.to(c.device)
            label = label.to(c.device)
            with torch.no_grad():
                e = encoder.encode_image(image).detach()
                e = e.to(c.device)

            # image = image.to(c.device)  # single scale
            # train decoder
            e_list = list()
            c_list = list()
            B, C = e.size()

            z, log_jac_det = decoders(e.to(dtype=torch.float32))

            decoder_log_prob = get_logp(C, z, log_jac_det)
            log_prob = decoder_log_prob / C  # likelihood per dim
            # loss = torch.sum(z**2,dim=(1,))/C
            loss = -(1.0-label) * log_theta(log_prob) - label * torch.log(1.0 - theta(log_prob))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # print(l,f)
            train_loss += t2np(loss.sum())
            train_count += len(loss)

        mean_train_loss = train_loss / train_count
        # run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        # save_weights(encoder, decoders, c.model, run_date)

        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
    #


def test_meta_epoch(c, epoch, loader, encoder, decoders, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = decoders.eval()

    image_list = list()
    gt_label_list = list()
    test_loss = list()
    test_z = list()
    test_count = 0
    norm_score = list()

    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=False)):
            # save
            if c.viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))

            # data
            image = image.to(c.device)
            e = encoder.encode_image(image).detach()  # BxCxHxW
            e = e.to(c.device)
  # single scale
            # test decoder
            e_list = list()

            B, C = e.size()
            z, log_jac_det = decoders(e.to(dtype=torch.float32))
            decoder_log_prob = get_logp(C, z, log_jac_det)
            score = -1 * torch.sum(z ** 2, dim=(1,))/C - 0* log_jac_det
            log_prob = decoder_log_prob / C  # likelihood per dim

            loss = -log_theta(log_prob)
            norm_score.append(1-theta(log_prob))

            test_z.append(z)
            test_loss.append(t2np(loss))

        z_grouped = torch.cat(norm_score, dim=0)
        anomaly_score = t2np(z_grouped)
        # z_grouped = torch.cat(test_z, dim=0)
        # anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(1,)))
        # score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
        #                      print_score=c.verbose or epoch == c.meta_epochs - 1)

    return image_list, gt_label_list, test_z, test_loss, anomaly_score


def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    encoder, preprocess = clip.load("ViT-L/14", device=c.device)
    # encoder, preprocess = clip.load("RN50", device=c.device)
    # print(encoder)
    # NF decoder
    decoders = load_decoder_arch(c, 768)
    total_params = sum(p.numel() for p in decoders.parameters() if p.requires_grad)
    print(total_params)
    decoders = decoders.to(c.device)

    # decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    # decoders = [decoder.to(c.device) for decoder in decoders]
    params = decoders.parameters()

    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    if c.dataset == 'mvtec':
        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset = StcDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False,
                                              **kwargs)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))
    # stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    print(c.norm_mean,c.norm_std)
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights(encoder, decoders, c.checkpoint)
            # pass
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            if(epoch==0):
                load_weights(encoder, decoders, c.checkpoint)
            train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))

        # height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_fps(
        #    c, epoch, test_loader, encoder, decoders, pool_layers, N)

        image_list, gt_label_list, test_z, test_loss, anomaly_score = test_meta_epoch(
            c, epoch, test_loader, encoder, decoders, N)

        is_anomaly = np.array([0 if l == 0 else 1 for l in gt_label_list])
        det_roc_obs = roc_auc_score(is_anomaly, anomaly_score)
        ap = average_precision_score(is_anomaly, anomaly_score)
        best_thres = find_best_threshold(is_anomaly, anomaly_score)
        best_acc = accuracy_score(is_anomaly, anomaly_score>best_thres)

        seg_pro_obs = 0
        seg_roc_obs = 0
        # print(anomaly_score)
        print(det_roc_obs)
        print(ap,best_thres, best_acc)
        run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # PxEHW
        save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
        save_weights(encoder, decoders, c.model, run_date)