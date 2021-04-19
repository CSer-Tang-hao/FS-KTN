# -*-coding:utf-8-*-
import os
import json
import torch
import random
import argparse
import torch.nn as nn
import os.path as osp
import numpy as np
from utils import check_dir
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss, cosine_similarity
# 引入GCN模型
from Models.GCN import GCN


def save_checkpoint(name):
    # torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '_fc' + '.pred'))

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=30,
                    help='number of training epochs')
parser.add_argument('--save_path', default='./experiments')
parser.add_argument('--network', type=str, default='Conv64',
                    help='choose which embedding network to use')
parser.add_argument('--dataset', type=str, default='miniImageNet',
                    help='choose which classification head to use. miniImageNet, tieredImageNet')
parser.add_argument('--trainval', default='10,0')
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--no-pred', action='store_true')

params = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('using gpu:', '0')
save_path = os.path.join(params.save_path, params.dataset, params.network, 'GCN')
check_dir(save_path)
graph = json.load(open('./mini-graph/fewshot-induced-graph.json', 'r'))
wnids, edges = graph['wnids'], graph['edges']

edges = edges + [(v, u) for (u, v) in edges]
edges = edges + [(u, u) for u in range(len(wnids))]# len(edges):32324, len(edges):97412

fc_vectors = torch.load('./experiments/miniImageNet/Conv64/1_stage_best_model.pth')['classifier']['weight_base']
fc_vectors = torch.cuda.FloatTensor(fc_vectors)
fc_vectors = F.normalize(fc_vectors)

word_vectors = torch.Tensor(graph['vectors']).cuda()
word_vectors = F.normalize(word_vectors)

hidden_layers = 'd1600,d'
gcn = GCN(len(wnids), edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()
print('{} nodes, {} edges'.format(len(wnids), len(edges)))
print('word vectors:', word_vectors.shape)
print('fc vectors:', fc_vectors.shape)
print('hidden layers:', hidden_layers)

optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001, weight_decay=0.0005)

v_train, v_val = map(float, params.trainval.split(','))
n_trainval = len(fc_vectors)
n_train = int(n_trainval * (v_train / (v_train + v_val)))
print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
tlist = list(range(len(fc_vectors)))
random.shuffle(tlist)

min_loss = 1e18

trlog = {}
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['min_loss'] = 0

labels = np.arange(64)
labels = torch.LongTensor(labels).cuda()

for epoch in range(1, params.max_epoch + 1):

    gcn.train()
    output_vectors = gcn(word_vectors)
    cls_scores = 10 * cosine_similarity(output_vectors,fc_vectors.transpose(1, 0),tlist[:n_train])
    cosloss = nn.CrossEntropyLoss()
    loss = cosloss(cls_scores,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    gcn.eval()

    output_vectors = gcn(word_vectors)
    train_loss = cosloss(cls_scores, labels).item()

    if v_val > 0:
        val_loss = cosloss(output_vectors, fc_vectors, tlist[n_train:]).item()
        loss = val_loss
    else:
        val_loss = 0
        loss = train_loss

    print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
          .format(epoch, train_loss, val_loss))

    trlog['train_loss'].append(train_loss)
    trlog['val_loss'].append(val_loss)
    trlog['min_loss'] = min_loss
    torch.save(trlog, osp.join(save_path, 'trlog'))

    if (epoch % params.save_epoch == 0):
        if params.no_pred:
            pred_obj = None
        else:
            pred_obj = {
                'wnids': wnids,
                'pred': output_vectors
            }

    if epoch % params.save_epoch == 0:
        save_checkpoint('epoch-{}'.format(epoch))

    pred_obj = None
