#在train_vitliu基础上修改，在clip提取特征之后接一个全连接层降维，查看参数数量
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
import matplotlib.pyplot as plt
OUT_DIR = './viz/'
WEIGHT_DIR = './weights'
gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

def cal_accuracy(y_true, y_pred, thres):
    N = y_true.shape[0]
    temp = deepcopy(y_pred)
    temp = [1 if i >= thres else 0 for i in temp]
    acc = (temp == y_true).sum() / N
    return acc

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    # if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
    #     return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        # temp[temp >= thres] = 1
        # temp[temp < thres] = 0
        temp=[1 if i>=thres else 0 for i in temp]
        # print(temp)
        acc = (temp == y_true).sum() / N
        # acc=accuracy_score(y_true, temp > thres)

        # print(acc)
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
            # print(best_thres,best_acc)

    return best_thres

def saveweights(encoder, MLP,decoders, model_name, run_date):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    state = {'encoder_state_dict': encoder.state_dict(),
             'MLP_state_dict': MLP.state_dict(),
             'decoder_state_dict': decoders.state_dict()}
    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(filename))


def loadweights(encoder,MLP, decoders, filename):
    path = os.path.join(filename)
    # state = torch.load(path, map_location='cuda:1')
    state=torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'], strict=True)
    MLP.load_state_dict(state['MLP_state_dict'],strict=True)
    decoders.load_state_dict(state['decoder_state_dict'], strict=True)
    print('Loading weights from {}'.format(filename))

def train_meta_epoch(c, epoch, loader, encoder, MLP,decoders, optimizer, N):
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
            e=MLP(e.to(dtype=torch.float32))
            B, C = e.size()


            z, log_jac_det = decoders(e.to(dtype=torch.float32))

            decoder_log_prob = get_logp(C, z, log_jac_det)
            log_prob = decoder_log_prob / C  # likelihood per dim
            # loss = torch.sum(z**2,dim=(1,))/C
            # loss = -(1.0 - label) * log_theta(log_prob) - 0.01 * (1.0 - label) * log_prob - label * torch.log(
            #     1.0 - theta(log_prob)) + 0.01 * label * log_prob
            loss=1./56.*label*log_prob - label * torch.log( 1.0 - theta(log_prob))

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


def test_meta_epoch(c, epoch, loader, encoder,MLP, decoders, N):
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

            e = MLP(e.to(dtype=torch.float32))
            e = e/torch.norm(e,p=2, dim=1,keepdim=True)
            # e = nn.SELU()(e)
            # test decoder
            e_list = list()

            B, C = e.size()
            z, log_jac_det = decoders(e.to(dtype=torch.float32))
            decoder_log_prob = get_logp(C, z, log_jac_det)
            score = -1 * torch.sum(z ** 2, dim=(1,))/C - 0* log_jac_det
            log_prob = decoder_log_prob / C  # likelihood per dim

            loss = -log_theta(log_prob)
            # norm_score.append(1-theta(log_prob))
            norm_score.append(-log_prob)
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
    MLP = nn.Linear(768,128)
    MLP.to(c.device)

    decoders = load_decoder_arch(c, 128)
    decoders = decoders.to(c.device)

    # decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    # decoders = [decoder.to(c.device) for decoder in decoders]
    params = list(MLP.parameters()) + list(decoders.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print(total_params)
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # load_weights(encoder, decoders, c.checkpoint)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    # if c.dataset == 'mvtec':
    #     train_dataset = MVTecDataset(c, is_train=True)
    #     test_dataset = MVTecDataset(c, is_train=False)
    # elif c.dataset == 'stc':
    #     train_dataset = StcDataset(c, is_train=True)
    #     test_dataset = StcDataset(c, is_train=False)
    # else:
    #     raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    loadweights(encoder, MLP, decoders, c.checkpoint)
    classname=['progan','cyclegan','biggan','stylegan','gaugan','stargan','deepfake','seeingdark','san',
                'crn','imle','guided','ldm_200','ldm_200_cfg','ldm_100','glide_100_27','glide_50_27',
                'glide_100_10','dalle']
    AP=[]
    Best_acc=[]
    Best_acc1=[]
    for k in classname:

        c.action_type = 'norm-test'
        c.meta_epochs = 1
        c.class_name = k

        test_dataset = MVTecDataset(c, is_train=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False,
                                                  **kwargs)
        N = 256  # hyperparameter that increases batch size for the decoder model by N
        print(k)
        print('test loader length', len(test_loader.dataset))
        print('test loader batches', len(test_loader))
        # stats




        image_list, gt_label_list, test_z, test_loss, anomaly_score = test_meta_epoch(
                c, 0, test_loader, encoder,MLP, decoders, N)

        is_anomaly = np.array([0 if l == 0 else 1 for l in gt_label_list])
        anomaly_score = np.array(anomaly_score)
        print(anomaly_score)

            # normal_score=[anomaly_score[i] for i in range(len(anomaly_score)) if is_anomaly[i]==0]
            # plt.hist(normal_score, bins=100, edgecolor='black', alpha=0.7)
            # plt.savefig('normal.png')

            # hist, bins = np.histogram(normal_score, bins=100000, range=(0,1.0 ), density=True)
            # cdf = np.cumsum(hist) * np.diff(bins)  # 累加并乘以bin宽度
            # cdf = np.insert(cdf, 0, 0)
            # percent_95 = bins[np.searchsorted(cdf, 0.95)]
            # percent_90 = bins[np.searchsorted(cdf, 0.90)]
            # percent_85 = bins[np.searchsorted(cdf, 0.85)]
            # print(percent_95,percent_90,percent_85)
            #
            # bad_score = [anomaly_score[i] for i in range(len(anomaly_score)) if is_anomaly[i] == 1]
            # plt.hist(bad_score, bins=100, edgecolor='black', alpha=0.7)
            # plt.savefig('bad.png')

        # det_roc_obs = roc_auc_score(is_anomaly, anomaly_score)
        ap = average_precision_score(is_anomaly, anomaly_score)
        best_thres = find_best_threshold(is_anomaly, anomaly_score)
        best_acc = accuracy_score(is_anomaly, anomaly_score>best_thres)
        threshold1=24.123487
        best_acc1 = accuracy_score(is_anomaly, anomaly_score > threshold1)
        best_acc1 = cal_accuracy(is_anomaly, anomaly_score, threshold1)
            # best_acc2 = accuracy_score(is_anomaly, anomaly_score > 280)
            # best_acc3 = accuracy_score(is_anomaly, anomaly_score > 260)

        print(ap,best_thres, best_acc)
        print(best_acc1)
        AP.append(ap)
        Best_acc.append(best_acc)
        Best_acc1.append(best_acc1)

        run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # PxEHW
        # save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
        with open( 'result.txt', 'a') as f:
            f.write('{:<15}'.format(k)+': ' + str(round(ap * 100, 2))+'  '+str(round(best_acc*100, 2))+'('+str(round(best_thres, 5))+')'+'   '+
                    str(round(best_acc1*100,2))+'('+str(round(threshold1,5))+')'+'\n')

    with open('result.txt', 'a') as f:
        f.write('{:<15}'.format('total') + ': ' + str(round(sum(AP)/len(AP) * 100, 2)) + '  ' + str(round(sum(Best_acc)/len(Best_acc) * 100, 2)) + '     ' + str(
            round(sum(Best_acc1)/len(Best_acc1)*100, 2)) + '\n')