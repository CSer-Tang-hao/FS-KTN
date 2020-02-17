# -*-coding:utf-8-*-
import time

import os
import os.path as osp
import shutil

import numpy as np
import scipy.sparse as sp
import torch

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


def one_hot(indices, depth):

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

def top1accuracy_all(output, target,nKbase):

    preds_data = output.data.cpu()
    labels_test_data = target.data.cpu()
    nKbase = nKbase.data.cpu()

    base_ids = torch.nonzero(labels_test_data < nKbase).view(-1)
    novel_ids = torch.nonzero(labels_test_data >= nKbase).view(-1)

    preds_base = preds_data[base_ids, :]
    preds_novel = preds_data[novel_ids, :]

    AccuracyBoth = top1accuracy(preds_data, labels_test_data)
    AccuracyBase = top1accuracy(preds_base[:, :nKbase], labels_test_data[base_ids])
    AccuracyNovel = top1accuracy(preds_novel[:, nKbase:], (labels_test_data[novel_ids] - nKbase))

    return AccuracyBoth, AccuracyBase, AccuracyNovel

def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu {}'.format(gpu))


def pick_vectors(dic, wnids, is_tensor=False):
    o = next(iter(dic.values()))
    dim = len(o)
    ret = []
    for wnid in wnids:
        v = dic.get(wnid)
        if v is None:
            if not is_tensor:
                v = [0] * dim
            else:
                v = torch.zeros(dim)
        ret.append(v)
    if not is_tensor:
        return torch.FloatTensor(ret)
    else:
        return torch.stack(ret)


def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def cosine_similarity(a,b,mask):
    return torch.mm(a[mask],b[:,mask])

def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

