# -*-coding:utf-8-*-

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from Models.Conv_model import Conv64,Conv128
from torch.autograd import Variable
from utils import check_dir, log, Timer,top1accuracy, one_hot, top1accuracy_all
from Models.Classifiers import KTN_Classifier
from mini_dataloder import MiniImageNet,FewShotDataloader


parser = argparse.ArgumentParser()
#********************* Model/Dataset/Path Config *********************#
parser.add_argument('--num_epoch', type=int, default=60,
                    help='number of training epochs')
parser.add_argument('--save_path', default='./experiments')
parser.add_argument('--network', type=str, default='Conv64',
                    help='choose which embedding network (Conv64/128) to use')
parser.add_argument('--dataset', type=str, default='miniImageNet',
                    help='choose which classification head to use. miniImageNet, tieredImageNet')
#********************* Training Config *********************#
parser.add_argument('--train_nKnovel', type=int, default=0, help='number of novel categories during 1-stage_base training phase')
parser.add_argument('--train_nKbase', type=int, default=64, help='number of base categories during 1-stage_base training phase')
parser.add_argument('--train_nExemplars', type=int, default=0, help='number of support examples per novel category')
parser.add_argument('--train_nTestNovel', type=int, default=0, help='number of query examples for all the novel category')
parser.add_argument('--train_nTestBase', type=int, default=32, help='number of test examples for all the base category')
parser.add_argument('--train_batch_size', type=int, default=8, help='number of episodes per batch')
parser.add_argument('--train_epoch_size', type=int, default=8*1000, help='number of batches per epoch')

#********************* Valing Config *********************#
parser.add_argument('--val_nKnovel', type=int, default=5, help='number of novel categories during 1-stage_base valing phase')
parser.add_argument('--val_nKbase', type=int, default=64, help='number of base categories during 1-stage_base valing phase')
parser.add_argument('--val_nExemplars', type=int, default=1, help='number of support examples per novel category')
parser.add_argument('--val_nTestNovel', type=int, default=15*5, help='number of query examples for all the novel category')
parser.add_argument('--val_nTestBase', type=int, default=15*5, help='number of test examples for all the base category')
parser.add_argument('--val_batch_size', type=int, default=1, help='number of episodes per batch')
parser.add_argument('--val_epoch_size', type=int, default=2000, help='number of batches per epoch')

params = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('using gpu:', '0')

save_path = os.path.join(params.save_path, params.dataset, params.network)
check_dir(save_path)

log_file_path = os.path.join(save_path, "train_log.txt")
log(log_file_path, str(vars(params)))

dataset_train = MiniImageNet(phase='train')
dataset_val = MiniImageNet(phase='val')

dataloder_train = FewShotDataloader(
    dataset=dataset_train,
    nKnovel=params.train_nKnovel,
    nKbase=params.train_nKbase,
    nExemplars=params.train_nExemplars,
    nTestNovel=params.train_nTestNovel,
    nTestBase=params.train_nTestBase,
    batch_size=params.train_batch_size,
    num_workers= 8,
    epoch_size=params.train_epoch_size,
)
dataloder_val = FewShotDataloader(
    dataset=dataset_val,
    nKnovel=params.val_nKnovel,
    nKbase=params.val_nKbase,
    nExemplars=params.val_nExemplars,
    nTestNovel=params.val_nTestNovel,
    nTestBase=params.val_nTestBase,
    batch_size=params.val_batch_size,
    num_workers= 8,
    epoch_size=params.val_epoch_size,
)
if params.network == 'Conv64':
    embedding_model = Conv64().cuda()
    classifier = KTN_Classifier(nKall=64, nFeat=64 * 5 * 5, scale_cls=10).cuda()
elif params.network == 'Conv128':
    embedding_model = Conv128().cuda()
    classifier = KTN_Classifier(nKall=64, nFeat=128 * 5 * 5, scale_cls=10).cuda()
else:  # the other backbones are coming soon!
    raise ValueError('Unknown models')

optimizer = torch.optim.SGD([{'params': embedding_model.parameters()},{'params': classifier.parameters()}],
                            lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

max_val_acc = 0.0
max_val_epoch = 0.0

timer = Timer()
Loss = torch.nn.CrossEntropyLoss()

for epoch in range(1, params.num_epoch + 1):
    # lr_scheduler.step()
    epoch_learning_rate = 0.1
    for param_group in optimizer.param_groups:
        epoch_learning_rate = param_group['lr']

    log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, epoch_learning_rate))

    _, _ = [x.train() for x in (embedding_model, classifier)]
    train_accuracies = []
    train_losses = []

    for i, batch in enumerate(tqdm(dataloder_train(epoch)), 1):

        data_train, labels_train = batch[0].cuda(), batch[1].view(params.train_batch_size*params.train_nTestBase).cuda()
        all_Kids, nKbase = batch[2], batch[3].squeeze()[0]

        emb_data_train = embedding_model(data_train.reshape([-1] + list(data_train.shape[-3:])))
        emb_data_train = emb_data_train.reshape(params.train_batch_size,params.train_nTestBase,-1)

        Kbase_ids =  Variable(all_Kids[:, :nKbase].contiguous(), requires_grad=False)

        cls_scores = classifier(features_query=emb_data_train, Kbase_ids=Kbase_ids).view(params.train_batch_size*params.train_nTestBase,-1)

        loss = Loss(cls_scores,labels_train)
        acc= top1accuracy(cls_scores.data, labels_train.data)

        train_accuracies.append(acc.item())
        train_losses.append(loss.item())

        if (i % 200 == 0):
            train_acc_avg = np.mean(np.array(train_accuracies))
            log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracyBase: {:.2f} % ({:.2f} %)'.format(
                epoch, i, len(dataloder_train), loss.item(), train_acc_avg, acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step()

    _, _ = [x.eval() for x in (embedding_model, classifier)]

    val_accuracies_both = []
    val_accuracies_base = []
    val_accuracies_novel = []

    for i, batch in enumerate(tqdm(dataloder_val(epoch)), 1):

        data_support, labels_support, data_query, labels_query, all_Kids, nKbase = [x.cuda() for x in batch]

        labels_support_one_hot = one_hot(labels_support.reshape(-1)-64,5).unsqueeze(dim=0)

        Kbase_ids = Variable(all_Kids[:, :nKbase].contiguous(), requires_grad=False)

        emb_data_support = embedding_model(data_support.view([-1] + list(data_support.shape[-3:])))
        emb_data_support = emb_data_support.view(params.val_batch_size,params.val_nKnovel,-1)

        emb_data_query = embedding_model(data_query.view([-1] + list(data_query.shape[-3:])))
        emb_data_query = emb_data_query.view(params.val_batch_size, params.val_nTestBase + params.val_nTestNovel, -1)

        cls_scores = classifier(features_query=emb_data_query, Kbase_ids=Kbase_ids,
                                     features_support=emb_data_support, labels_support=labels_support_one_hot).view(
                                           params.val_batch_size*(params.val_nTestBase+params.val_nTestNovel),-1)

        accuracyBoth, accuracyBase, accuracyNovel = top1accuracy_all(cls_scores, labels_query.view(-1), nKbase)

        val_accuracies_both.append(accuracyBoth.item())
        val_accuracies_base.append(accuracyBase.item())
        val_accuracies_novel.append(accuracyNovel.item())

    val_acc_both = np.mean(np.array(val_accuracies_both))
    val_acc_both_ci95 = 1.96 * np.std(np.array(val_accuracies_both)) / np.sqrt(params.val_epoch_size)

    val_acc_base = np.mean(np.array(val_accuracies_base))
    val_acc_base_ci95 = 1.96 * np.std(np.array(val_accuracies_base)) / np.sqrt(params.val_epoch_size)

    val_acc_novel = np.mean(np.array(val_accuracies_novel))
    val_acc_novel_ci95 = 1.96 * np.std(np.array(val_accuracies_novel)) / np.sqrt(params.val_epoch_size)

    if val_acc_novel > max_val_acc:
        max_val_acc = val_acc_novel
        max_val_epoch = epoch
        torch.save({'embedding': embedding_model.state_dict(), 'classifier': classifier.state_dict()}, os.path.join(save_path, '1_stage_best_model.pth'))
        log(log_file_path, 'Best model saving!!!')
        log(log_file_path,
            'Val_Epoch: [{}/{}]\tAccuracyBoth: {:.2f} +- {:.2f} %\tAccuracyBase: {:.2f} +- {:.2f} %\tAccuracyNovel: {:.2f} +- {:.2f} %'.format(
            epoch, params.num_epoch,
            val_acc_both, val_acc_both_ci95,
            val_acc_base, val_acc_base_ci95,
            val_acc_novel, val_acc_novel_ci95))
    else:
        log(log_file_path,
            'Val_Epoch: [{}/{}]\tAccuracyBoth: {:.2f} +- {:.2f} %\tAccuracyBase: {:.2f} +- {:.2f} %\tAccuracyNovel: {:.2f} +- {:.2f} %'.format(
                epoch, params.num_epoch,
                val_acc_both, val_acc_both_ci95,
                val_acc_base, val_acc_base_ci95,
                val_acc_novel, val_acc_novel_ci95))

    torch.save({'embedding': embedding_model.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(save_path, '1_stage_last_model.pth'))

    log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(params.num_epoch))))

log(log_file_path,
    'Best model saving!!!\tBest_Epoch: [{}/{}]\tAccuracy_Val_Novel: {:.2f} %'.format(
    max_val_epoch, params.num_epoch,max_val_acc))