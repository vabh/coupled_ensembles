from __future__ import print_function
import argparse
import os
import random
import setproctitle
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import logger
from coupled_ensemble import CoupledEnsemble


def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', help='cifar10 | cifar100 | cifar20 | joint | fold')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--microBatch', type=int, default=64, help='process data in reduced batch size for large models')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
    parser.add_argument('--weightDecay', type=float, default=0.0001, help='weight decay, default=0.0001')
    parser.add_argument('--cuda', action='store_false', help='enables cuda')
    parser.add_argument('--save', help='folder to store log files, model checkpoints')
    parser.add_argument('--resume', help='checkpoint file to resume training from')
    parser.add_argument('--startEpoch', type=int, default=0, help='epoch number to start training from')
    parser.add_argument('--manualSeed', type=int, default=-1)
    parser.add_argument('--nGPU', type=int, default=1)

    parser.add_argument('--k', type=int, default=12)
    parser.add_argument('--L', type=int, default=100)
    parser.add_argument('--num', type=int, default=4)

    parser.add_argument('--probs', action='store_true', help='To choose CELoss or NLLLoss')

    return parser


# get data loaders
def get_data_loaders(opt):
    cifar100_normTransform = transforms.Normalize(
                                (129.3/255,124.1/255,112.4/255),
                                (68.2/255,65.4/255,70.4/255))

    print('Dataset: ' + opt.dataset)
    if opt.dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=True,
                               transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
        test_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                               transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
    elif opt.dataset == 'cifar100':
        train_dataset = dset.CIFAR100(root=opt.dataroot, download=True, train=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   cifar100_normTransform,
                               ]))
        test_dataset = dset.CIFAR100(root=opt.dataroot, download=True, train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   cifar100_normTransform,
                               ]))
    elif opt.dataset == 'svhn':
        train_dataset = dset.SVHN(root=opt.dataroot, download=True, split='train',
                               transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
        train_extra_dataset = dset.SVHN(root=opt.dataroot, download=True, split='extra',
                               transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
        from dataset import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, train_extra_dataset])
        test_dataset = dset.SVHN(root=opt.dataroot, download=True, split='test',
                               transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
    assert train_dataset
    assert test_dataset

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.microBatch,
                                               shuffle=True, num_workers=int(opt.workers))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.microBatch,
                                              shuffle=False, num_workers=int(opt.workers))
    return train_loader, test_loader


def get_model(opt):
    nClasses = 0
    if opt.dataset == 'cifar10':
        nClasses = 10
    elif opt.dataset == 'cifar100':
        nClasses = 100
    elif opt.dataset == 'svhn':
        nClasses = 10
    assert nClasses != 0

    
    probs = opt.probs
    dropout = False
    if opt.dataset == 'svhn':
        dropout = True
        opt.niter = 40

    net = CoupledEnsemble(None, opt.num, probs, ensemble=True, k=opt.k, L=opt.L, dropout=dropout, nClasses=nClasses)

    nParams = 0
    for p in net.parameters():
        nParams += p.data.nelement()
    print('#Params: ', nParams)
    # print(net)

    # criterion
    if probs is True:
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    print('Loss:', type(criterion).__name__)

    if opt.cuda:
        net.cuda()
        criterion.cuda()
    if opt.nGPU > 1:
        net = nn.DataParallel(net).cuda()

    return net, criterion


def train(epoch, net, criterion, train_loader, optimizer, opt):
    # set mode
    net.train()

    lr = adjust_learning_rate(opt, optimizer, epoch)

    score_epoch = 0
    loss_epoch = 0
    optim_batch = 0  # micro batch training
    UPDATE_EVERY = opt.batchSize // opt.microBatch  # process microBatch images, update batchSize grads
    start = time.time()
    data_start = time.time()
    data_time = 0
    print('\nEpoch: %d LR: %.4f %d' % (epoch, lr, UPDATE_EVERY))
    for i, (images, labels) in enumerate(train_loader):
        data_time += time.time() - data_start

        if opt.dataset == 'svhn':
            labels = labels.long() - 1
            labels = labels.squeeze()

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        out = net(images)
        loss = criterion(out, labels)
        loss.backward()

        optim_batch += 1
        if optim_batch == UPDATE_EVERY:
            optim_batch = 0
            optimizer.step()
            optimizer.zero_grad()

        loss_epoch = loss_epoch + loss.data[0]
        score_epoch = score_epoch + compute_score(out.data, labels.data)

        data_start = time.time()

    print('Total time: {0:.4f}, Data: {1:.4f}' .format((time.time() - start), data_time*1.0 / i))
    loss_epoch = loss_epoch / len(train_loader)
    print('[%d/%d][%d] train_loss: %.4f err: %d'
          % (epoch, opt.niter, len(train_loader), loss_epoch, score_epoch))

    return loss_epoch, score_epoch


# test network
def test(net, criterion, test_loader, opt):
    net.eval()
    score_epoch = 0
    loss_epoch = 0
    start = time.time()
    data_start = time.time()
    data_time = 0
    for i, (images, labels) in enumerate(test_loader):
        data_time += time.time() - data_start

        if opt.dataset == 'fold':
            labels = labels[:, 1]

        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels).cuda()

        out = net(images)
        loss = criterion(out, labels)

        loss_epoch += loss.data[0]
        score_epoch = score_epoch + compute_score(out.data, labels.data)

        data_start = time.time()

    loss_epoch /= len(test_loader)
    print('Total time: {0:.4f}, Data: {1:.4f}' .format((time.time() - start), data_time*1.0 / i))
    print('Test err: %d, Loss: %.4f' % (score_epoch, loss_epoch))

    return score_epoch


def main():

    # get command line args
    parser = setup_args()
    opt = parser.parse_args()
    print(opt)

    # logger
    setproctitle.setproctitle(opt.save)
    try:
        os.makedirs(opt.save)
        print('Logging at: ' + opt.save)
    except OSError:
        pass
    torch.save(opt, os.path.join(opt.save, 'opt.pth'))
    log_path = os.path.join(opt.save, 'train.log')
    log = logger.Logger(log_path, ['loss', 'train_error', 'test_error'])

    # set random seed
    if opt.manualSeed == -1:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.cuda:
        torch.cuda.manual_seed(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    # perforamnce options
    cudnn.benchmark = True
    # torch.set_num_threads(8)

    # init model
    net, criterion = get_model(opt)
    # optimizer options
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9,
                          weight_decay=opt.weightDecay, nesterov=True)

    start_epoch = opt.startEpoch
    best_error = 9999999999
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    train_loader, test_loader = get_data_loaders(opt)
    _nTrain = len(train_loader.dataset)*1.0
    _nTest = len(test_loader.dataset)*1.0
    print('Train samples: ', _nTrain)
    print('Test samples: ', _nTest)

    # test error on model init
    test_error = test(net, criterion, test_loader, opt)

    # train for opt.niter epochs
    for epoch in range(start_epoch, opt.niter):

        loss, train_error = train(epoch, net, criterion, train_loader,
                                  optimizer, opt)
        if epoch >= opt.niter - 300:
            test_error = test(net, criterion, test_loader, opt)
        else:
            test_error = -1
        log.add([loss, train_error/_nTrain, test_error/_nTest])
        log.plot()

        is_best = test_error < best_error
        best_error = min(test_error, best_error)
        # do checkpointing

        _checkpoint_dict = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_error': best_error}
        save_checkpoint(_checkpoint_dict,
                        filename=os.path.join(opt.save, 'latest.pth'))
        if is_best:
            save_checkpoint(_checkpoint_dict,
                            filename=os.path.join(opt.save, 'net_best.pth'))

        if epoch % 10 == 0 or epoch >= (opt.niter-10):
            save_checkpoint(_checkpoint_dict,
                            filename=os.path.join(opt.save, 'net_epoch_%d.pth' % (epoch)))


# count number of incorrect classifications
def compute_score(output, target):
    pred = output.max(1)[1]
    incorrect = pred.ne(target).cpu().sum()
    return incorrect


def adjust_learning_rate(opt, optimizer, epoch):
    if epoch >= 0.75*opt.niter:
        lr = opt.lr * 0.01
    elif epoch >= 0.5*opt.niter:
        lr = opt.lr * 0.1
    else:
        lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
