# -*-coding:utf-8-*-
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from Models.Conv_model import Conv64,Conv128
from torch.autograd import Variable
from utils import one_hot, top1accuracy_all
from Models.Classifiers import KTN_Classifier
from mini_dataloder import MiniImageNet,FewShotDataloader

parser = argparse.ArgumentParser()

parser.add_argument('--num_epoch', type=int, default=1,
                    help='number of training epochs')
parser.add_argument('--save_path', default='./experiments')
parser.add_argument('--network', type=str, default='Conv64',
                    help='choose which embedding network to use')
parser.add_argument('--dataset', type=str, default='miniImageNet',
                    help='choose which classification head to use. miniImageNet, tieredImageNet')

#***********************************************************************************************
parser.add_argument('--test_nKnovel', type=int, default=5, help='number of novel categories during 1-stage_base testing phase')
parser.add_argument('--test_nKbase', type=int, default=64, help='number of base categories during 1-stage_base testing phase')
parser.add_argument('--test_nExemplars', type=int, default=1, help='number of support examples per novel category')
parser.add_argument('--test_nTestNovel', type=int, default=15*5, help='number of query examples for all the novel category')
parser.add_argument('--test_nTestBase', type=int, default=15*5, help='number of test examples for all the base category')
parser.add_argument('--test_batch_size', type=int, default=1, help='number of episodes per batch')
parser.add_argument('--test_epoch_size', type=int, default=600, help='number of batchs per epoch')

params = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('using gpu:', '0')

save_path = os.path.join(params.save_path, params.dataset, params.network)


dataset_test = MiniImageNet(phase='test')
dataloder_test = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=params.test_nKnovel,
    nKbase=params.test_nKbase,
    nExemplars=params.test_nExemplars,
    nTestNovel=params.test_nTestNovel,
    nTestBase=params.test_nTestBase,
    batch_size=params.test_batch_size,
    num_workers= 4,
    epoch_size=params.test_epoch_size,
)

if params.network == 'Conv64':
    embedding_model = Conv64().cuda()
    classifier = KTN_Classifier(nKall=64, nFeat=64 * 5 * 5, scale_cls=10).cuda()
else:  # the other backbones are coming soon!
    raise ValueError('Unknown models')

best_model = torch.load(os.path.join(save_path,'1_stage_best_model.pth'))

embedding_model.load_state_dict(best_model['embedding'])
classifier.load_state_dict(best_model['classifier'])

for epoch in range(5,15):
    # Only for Conv64 Model
    print('Loading epoch-{}_fc.pred'.format(epoch))
    pred_fcw = torch.load('./experiments/miniImageNet/Conv64/GCN/epoch-{}_fc.pred'.format(epoch))

    _, _ = [x.eval() for x in (embedding_model, classifier)]

    test_accuracies_both = []
    test_accuracies_base = []
    test_accuracies_novel = []

    for i, batch in enumerate(tqdm(dataloder_test()), 1):

        data_support, labels_support, data_query, labels_query, all_Kids, nKbase = [x.cuda() for x in batch]

        labels_support_one_hot = one_hot(labels_support.reshape(-1) - 64, 5).unsqueeze(dim=0)

        Kbase_ids = Variable(all_Kids[:, :nKbase].contiguous(), requires_grad=False)
        Knovel_ids = Variable(all_Kids[:, nKbase:].contiguous(), requires_grad=False)

        emb_data_support = embedding_model(data_support.view([-1] + list(data_support.shape[-3:])))
        emb_data_support = emb_data_support.view(params.test_batch_size, params.test_nExemplars*params.test_nKnovel, -1)

        emb_data_query = embedding_model(data_query.view([-1] + list(data_query.shape[-3:])))
        emb_data_query = emb_data_query.view(params.test_batch_size, params.test_nTestBase + params.test_nTestNovel,-1)

        cls_scores = classifier(features_query=emb_data_query, Kbase_ids=Kbase_ids,
                                Knovel_ids = Knovel_ids,pred_fcw = pred_fcw,
                                features_support=emb_data_support, labels_support=labels_support_one_hot).view(
            params.test_batch_size * (params.test_nTestBase + params.test_nTestNovel), -1)

        accuracyBoth, accuracyBase, accuracyNovel = top1accuracy_all(cls_scores, labels_query.view(-1), nKbase)

        test_accuracies_both.append(accuracyBoth.item())
        test_accuracies_base.append(accuracyBase.item())
        test_accuracies_novel.append(accuracyNovel.item())

    test_acc_both = np.mean(np.array(test_accuracies_both))
    test_acc_both_ci95 = 1.96 * np.std(np.array(test_accuracies_both)) / np.sqrt(params.test_epoch_size)

    test_acc_base = np.mean(np.array(test_accuracies_base))
    test_acc_base_ci95 = 1.96 * np.std(np.array(test_accuracies_base)) / np.sqrt(params.test_epoch_size)

    test_acc_novel = np.mean(np.array(test_accuracies_novel))
    test_acc_novel_ci95 = 1.96 * np.std(np.array(test_accuracies_novel)) / np.sqrt(params.test_epoch_size)

    print('AccuracyBoth: {:.2f} +- {:.2f} %\tAccuracyBase: {:.2f} +- {:.2f} %\tAccuracyNovel: {:.2f} +- {:.2f} %'.format(
                test_acc_both, test_acc_both_ci95, test_acc_base, test_acc_base_ci95,test_acc_novel, test_acc_novel_ci95))

