import datetime
import json
import os
import setproctitle
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from resnet import ResNet152
from cmd_options import get_arguments


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batchSz', type=int, default=64)
    # parser.add_argument('--nEpochs', type=int, default=300)
    # parser.add_argument('--no-cuda', action='store_true')
    # parser.add_argument('--save')
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--opt', type=str, default='sgd',
    #                     choices=('sgd', 'adam', 'rmsprop'))
    # args = parser.parse_args()
    #
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.save = args.save or 'work/densenet.base'
    # setproctitle.setproctitle(args.save)
    #
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    date = datetime.datetime.now().strftime("[%m-%d-%H:%M]")
    args = get_arguments()
    args.batchSz = args.batch_size
    args.nEpochs = args.n_epochs
    args.opt = args.optimizer

    args.cuda = args.gpus and torch.cuda.is_available()
    setproctitle.setproctitle(args.name)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)


    if args.augment == True:
        aug_flag = True
        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])
    else:
        aug_flag = False
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])
    print("Using augmentation" if aug_flag else "Diable augmentation")

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    if args.dataset == "cifar10":
        dataset = dset.CIFAR10
        nClasses = 10
    elif args.dataset == "cifar100":
        dataset = dset.CIFAR100
        nClasses = 100

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dataset(root='cifar', train=True, download=True,
                      transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dataset(root='cifar', train=False, download=True,
                      transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = ResNet152()
    tname = 'ResNet152'

    ####################### dump configs to file #######################
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    with open(os.path.join(args.save, 'config.txt'), 'w+') as fp:
        fp.write("Using augmentation\n" if aug_flag else "Diable augmentation\n")
        fp.write('{}  : Number of params: {} \n'.format(
            tname,
            sum([p.data.nelement() for p in net.parameters()]))
        )
        fp.write(json.dumps(vars(args), indent=4) + "\n")
        fp.write(str(net))

    ############################# tf logger ##############################
    try:
        from tools.logger import Logger
    except ImportError as e:
        import time
        print("fail to import tensorboard: {} ".format(e))
        time.sleep(2)
    else:
        global g_logger
        g_logger = Logger(args.save, restart=False)

    if args.cuda:
        if len(args.gpus) > 1:
            print(args.gpus)
            net = nn.DataParallel(net ,device_ids=args.gpus)
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                              momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    best_error = 100

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch, args)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        loss, err = test(args, epoch, net, testLoader, optimizer, testF)
        os.system('./plot.py {} &'.format(args.save))
        if err < best_error:
            torch.save({
                "epoch": epoch,
                "isBest": True,
                "Error": err,
                "optimState": optimizer.state,
                "model": net
            }, os.path.join(args.save, 'best.pth'))

        torch.save({
            "epoch": epoch,
            "isBest": err < best_error,
            "Error": err,
            "optimState": optimizer.state,
            "model": net
        }, os.path.join(args.save, 'latest.pth'))

    trainF.close()
    testF.close()


def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100. * incorrect / len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('{} | Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            args.name, partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

        # ================= tensorboard visualization ===================
        if g_logger is not None:
            info = {
                'train-loss': loss.data[0],
                'train-top1-error': err
            }
            partialEpoch = epoch + batch_idx / len(trainLoader)
            for tag, value in info.items():
                g_logger.scalar_summary(tag, value, partialEpoch)


def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100. * incorrect / nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

    # ================= tensorboard visualization ===================
    if g_logger is not None:
        info = {
            'valid-loss': test_loss,
            'valid-top1-error': err
        }
        for tag, value in info.items():
            g_logger.scalar_summary(tag, value, epoch)

    return test_loss, err


def adjust_opt(optAlg, optimizer, epoch, args=None):
    nEpoch = 300
    if args is not None:
        nEpoch = args.n_epochs

    stage1 = int(nEpoch * 0.5 )
    stage2 = int(nEpoch * 0.75)

    if optAlg == 'sgd':
        if epoch < stage1:
            lr = 1e-1
        elif epoch == stage1:
            lr = 1e-2
        elif epoch == stage2:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    # tf logger
    g_logger = None
    main()
