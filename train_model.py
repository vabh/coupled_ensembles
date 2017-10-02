from __future__ import print_function
import argparse
import os
import random
import setproctitle
import time
import sys
import yaml
import numpy as np

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
    parser.add_argument('--configFile', help='Specify options to this script through a .yaml file')

    parser.add_argument('--dataset', default='cifar100', help='cifar10 | cifar100 | cifar20 | joint | fold')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--microBatch', type=int, default=64, help='process data in reduced batch size for large models')

    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--startEpoch', type=int, default=0, help='epoch number to start training from')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
    parser.add_argument('--lrFile', help='A txt file with each line corresponding to lr for that epoch. If give, this overrides --lr and --niter')
    parser.add_argument('--weightDecay', type=float, default=0.0001, help='weight decay, default=0.0001')
    parser.add_argument('--sgdMomentum', type=float, default=0.9, help='SGD mometum, default=0.9')
    parser.add_argument('--bnMomentum', type=float, default=0.1, help='BN momentum for running mean, default=0.1')

    parser.add_argument('--save', help='folder to store log files, model checkpoints')
    parser.add_argument('--saveN', type=int, default=10, help='save last N epochs')
    parser.add_argument('--resume', help='checkpoint file to resume training from')
    parser.add_argument('--testOnly', action='store_true', help='Test model on data and loaded weights')

    parser.add_argument('--manualSeed', type=int, default=-1)
    parser.add_argument('--cuda', action='store_false', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1)

    parser.add_argument('--arch', default='densenet', help='choose basic block architecture: densenet | resnet')
    parser.add_argument('--archConfig', help='Provide arch specific properties as "prop=val"')

    parser.add_argument('--E', type=int, default=4)
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
    elif opt.dataset == 'mnist':
        class FashionMNIST(dset.MNIST):
            """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
            """
            urls = [
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
            ]

        train_dataset = FashionMNIST(root=opt.dataroot, download=True, train=True,
                               transform=transforms.Compose([
                                   # transforms.RandomHorizontalFlip(),
                                   # transforms.RandomCrop(32, padding=4),
                                   transforms.Scale(32),
                                   transforms.ToTensor(),
                               ]))
        test_dataset = FashionMNIST(root=opt.dataroot, download=True, train=False,
                               transform=transforms.Compose([
                                   transforms.Scale(32),
                                   transforms.ToTensor(),
                               ]))
    assert train_dataset
    assert test_dataset

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.microBatch,
                                               shuffle=True, num_workers=int(opt.workers))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.microBatch,
                                              shuffle=False, num_workers=int(opt.workers))

    num_classes = 0
    if opt.dataset == 'cifar10':
        num_classes = 10
    elif opt.dataset == 'cifar100':
        num_classes = 100
    elif opt.dataset == 'svhn':
        num_classes = 10
    elif opt.dataset == 'mnist':
        num_classes = 10
    assert num_classes != 0

    return train_loader, test_loader, num_classes


def get_model(opt, arch_config):

    dropout = False
    if opt.dataset == 'svhn':
        dropout = True
        opt.niter = 40

    if not arch_config.has_key('dropout'):
        arch_config['dropout'] = dropout

    net = CoupledEnsemble(opt.arch, opt.E, opt.probs, ensemble=True,
                          **arch_config)


    # set BN momentum for running mean
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = opt.bnMomentum

    nParams = 0
    for p in net.parameters():
        nParams += p.data.nelement()
    print('#Params: ', nParams)

    # criterion
    if opt.probs is True:
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
    print('\nEpoch: [%d/%d] LR: %.4f' % (epoch+1, opt.niter, lr))
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

    loss_epoch = loss_epoch / len(train_loader)
    print('[Train] Time: {0:.4f}, Loss: {1:.4f} Err: {2:d}' .format((time.time() - start), loss_epoch, score_epoch))

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

        if opt.dataset == 'svhn':
            labels = labels.long() - 1
            labels.squeeze()

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
    print('[Test]  Time: %.4f, Loss: %.4f, Err: %d' % (time.time() - start, loss_epoch, score_epoch))

    return loss_epoch, score_epoch


def main():

    # get command line args
    parser = setup_args()
    opt = parser.parse_args()
    try:
        with open(opt.configFile, 'r') as f:
            yaml_params = yaml.safe_load(f)
        # shallow merge yaml params with opt
        for k, v in yaml_params.iteritems():
            try:
                # cmd arg overwrites the value from the yaml file
                flag = 0
                for passed_arg in sys.argv:
                    if k == passed_arg[2:]:
                        flag = 1
                        break
                if flag == 0:
                    setattr(opt, k, v)
            except:
                pass
    except:
        print("File not found or cannot be opened", opt.configFile)
    print(opt)
    if opt.lrFile is not None and os.path.isfile(opt.lrFile):
        opt.lrRates = np.loadtxt(opt.lrFile)
        opt.niter = len(opt.lrRates)
        print('Using LRs from "%s", training for: %d epochs' % (opt.lrFile, opt.niter))
    else:
        opt.lrRates = None

    # logger
    setproctitle.setproctitle(opt.save)
    try:
        os.makedirs(opt.save)
        print('Logging at: ' + opt.save)
    except OSError:
        pass
    torch.save(opt, os.path.join(opt.save, 'opt.pth'))
    log_path = os.path.join(opt.save, 'train.log')
    log = logger.Logger(log_path, ['loss', 'train_error', 'test_loss', 'test_error'])

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

    # data
    train_loader, test_loader, num_classes = get_data_loaders(opt)
    nTrain = len(train_loader.dataset)*1.0
    nTest = len(test_loader.dataset)*1.0
    print('Train samples: ', nTrain)
    print('Test samples: ', nTest)

    # init model
    # get architecutre specific options
    arch_config = {}
    try:
        arch_config_string = opt.archConfig
        params = map(lambda x: x.split('='), arch_config_string.split(','))
        for (k, v) in params:
            # TODO: handle datatypes, this is potentially tricky
            # YAML does auto converstion from string -> int, float, bool
            if v.isdigit():
                v = int(v)
            elif v == "True" or v == "False":
                if v == "True":
                    v = True
                else:
                    v = False
            else:
                v = float(v)
            arch_config[k.strip()] = v

        # print(arch_config)
    except:
        print("archConfig string received: ", arch_config_string)
        # raise ValueError

    arch_config['num_classes'] = num_classes
    net, criterion = get_model(opt, arch_config)
    # optimizer options
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.sgdMomentum,
                          weight_decay=opt.weightDecay, nesterov=True)

    # resume
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

    # test error on model init
    test_loss, test_error = test(net, criterion, test_loader, opt)
    if opt.testOnly:
        return

    if start_epoch == 0:
        log.add(['NaN', 'NaN', test_loss, test_error/nTest])

        # save the initial model state
        _checkpoint_dict = {
                'epoch': 0,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_error': test_error}
        save_checkpoint(_checkpoint_dict,
                        filename=os.path.join(opt.save, 'net_epoch_0.pth'))
        save_checkpoint(_checkpoint_dict,
                        filename=os.path.join(opt.save, 'latest.pth'))

    # train for opt.niter epochs
    for epoch in range(start_epoch, opt.niter):

        loss, train_error = train(epoch, net, criterion, train_loader,
                                  optimizer, opt)
        test_loss, test_error = test(net, criterion, test_loader, opt)
        log.add([loss, train_error/nTrain, test_loss, test_error/nTest])
        log.plot()

        # checkpointing
        is_best = test_error < best_error
        best_error = min(test_error, best_error)
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

        if (epoch+1) % 10 == 0 or epoch >= (opt.niter - opt.saveN):
            save_checkpoint(_checkpoint_dict,
                            filename=os.path.join(opt.save, 'net_epoch_%d.pth' % (epoch+1)))


# count number of incorrect classifications
def compute_score(output, target):
    pred = output.max(1)[1]
    incorrect = pred.ne(target).cpu().sum()
    return incorrect


def adjust_learning_rate(opt, optimizer, epoch):
    if opt.lrRates is not None:
        return opt.lrRates[epoch]

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
