# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

class LinearDiag(nn.Module):

    def __init__(self, num_features, bias=False):

        super(LinearDiag, self).__init__()

        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out

class FeatExemplarAvgBlock(nn.Module):

    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1,2)

        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))

        return weight_novel



class KTN_Classifier(nn.Module):

    def __init__(self,nKall=64, nFeat=64 * 5 * 5, scale_cls=10.0):
        super(KTN_Classifier, self).__init__()

        self.favgblock = FeatExemplarAvgBlock(nFeat)

        weight_base = torch.FloatTensor(nKall, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)

        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)

        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)


    def get_classification_weights(
            self, Kbase_ids, Knovel_ids, pred, features_support, labels_support):

        batch_size, nKbase = Kbase_ids.size()

        weight_base = self.weight_base[Kbase_ids.view(-1)]
        weight_base = weight_base.view(batch_size, nKbase, -1)

        if features_support is None or labels_support is None:

            return weight_base

        else:

            _, num_train_examples, num_channels = features_support.size()

            nKnovel = labels_support.size(2)

            features_support = F.normalize(
                    features_support, p=2, dim=features_support.dim() - 1, eps=1e-12)

            weight_novel = self.favgblock(features_support, labels_support)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)

            if pred is not None:

                fcw = torch.stack(
                    (pred['pred'][Knovel_ids[0][0]], pred['pred'][Knovel_ids[0][1]], pred['pred'][Knovel_ids[0][2]],
                     pred['pred'][Knovel_ids[0][3]], pred['pred'][Knovel_ids[0][4]]), 0)
                fcw = fcw.view(batch_size, nKnovel, num_channels)

                coefficient = float(5 / labels_support.size()[1])
                weight_novel = weight_novel + fcw * coefficient

            weight_both = torch.cat([weight_base, weight_novel], dim=1)

            return weight_both

    def apply_classification_weights(self, features, cls_weights):

        features = F.normalize(
                features, p=2, dim=features.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(
                cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0,self.bias.view(1, 1, 1), 1.0, features,
        cls_weights.transpose(1, 2))
        return cls_scores

    def forward(self, features_query=None, Kbase_ids=None, Knovel_ids=None, pred_fcw = None, features_support=None, labels_support=None):

        cls_weights = self.get_classification_weights(
            Kbase_ids, Knovel_ids, pred_fcw, features_support, labels_support)

        cls_scores = self.apply_classification_weights(
            features_query, cls_weights)

        return cls_scores
