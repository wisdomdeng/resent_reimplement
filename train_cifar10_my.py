## TODO: Training, Testing, Save model, logging, loading data, setting device, learning rate schedule, image_transformation

import tensorflow as tf

import torch
import torch.optim as optim
import time
from torch import tensor
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from resnet import ResNet152
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

GPUs = 4
BATCH_SIZE = 64
NUM_EPOCHS = 300
## Change GPUs to 0 when running locally

def train(net, data, loss, optimizer, epoch, totalEpoch, useCuda=False, logFile=None):
    net.train()
    numBatch = len(data)
    totalError = 0
    numTraining = len(data.dataset)
    totalLoss = 0
    for batchId, (image, label) in enumerate(data):
        if useCuda:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = net(image)
        batchLoss = loss(output, label)
        batchLoss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        error = torch.ne(pred.cpu().data, label.cpu().data).sum()
        totalError += error
        lossValue = float(batchLoss.data[0])
        totalLoss += lossValue
        errorRate = error/len(image)
        logString = 'Epoch {:d} of {:d} Training Batch {:d} of {:d} Training Error: {:.6f} ' \
                    'Training loss: {:.6f}\n'.format(epoch+1, totalEpoch, batchId+1, numBatch, errorRate, lossValue)
        print(logString)
        if logFile is not None:
            logFile.write(logString)
    totalErrorRate = totalError/numTraining
    avgLoss = totalLoss / numBatch
    logString = 'Epoch {:d} of {:d} Training Error: {:.6f} Training loss: {:.6f}\n'.format(epoch+1, totalEpoch, errorRate,
                                                                                           lossValue)
    print('')
    print(logString)
    if logFile is not None:
        logFile.write('\n')
        logFile.write(logString)
    return totalErrorRate, avgLoss

def test(net, data, loss, epoch, totalEpoch, useCuda=False, logFile=None):
    net.eval()
    ## Assume test images one by one
    totalLoss = 0
    totalError = 0
    numTest = len(data.dataset)
    numIteration = len(data)
    for image, label in data:
        if useCuda:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        output = net(image)
        sampleLoss = loss(output, label)
        pred = output.max(1)[1]
        error = torch.ne(pred.cpu().data, label.cpu().data).sum()
        totalError += error
        lossValue = float(sampleLoss.data[0])
        totalLoss += lossValue
    avgLoss = totalLoss/numIteration
    avgError = totalError/numTest
    logString = 'Epoch {:d} of {:d} Validation Error: {:.6f} Validation Loss {:.6f}\n\n'.format(epoch+1, totalEpoch,
                                                                                                avgError, avgLoss)
    print(logString)
    if logFile is not None:
        logFile.write(logString)
    return avgError, avgLoss

def sgd_lr_scheduler(opt, epoch, firstPoint=150, secondPoint=225):
    if epoch <= 1:
        lr = 1e-1
    elif epoch == firstPoint:
        lr = 1e-2
    elif epoch == secondPoint:
        lr = 1e-3
    else:
        return
    for group in optim.param_groups:
        group['lr'] = lr
    return lr

def make_plot(data, epoch, path, minEpoch=None, minError=None):
    train_errors, train_losses, val_errors, val_losses = data
    fig, ax = plt.subplot(1, 1, figsize=(6, 5))
    plt.plot(list(range(1, epoch+1)), train_losses, 'r-', label='Train Loss')
    plt.plot(list(range(1, epoch + 1)), val_losses, 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    ax.set_yscale('log')
    ax.set_title('Cross Entropy vs Training Epoch')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.pdf'))

    fig, ax = plt.subplot(1, 1, figsize=(6, 5))
    plt.plot(list(range(1, epoch + 1)), train_errors, 'r-', label='Train Errors')
    plt.plot(list(range(1, epoch + 1)), val_errors, 'b-', label='Validation Errors')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    ax.set_title('Error vs Training Epoch')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.pdf'))

def main():
    if torch.cuda.is_available() and GPUs > 0:
        useCuda=True
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)
    trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), normTransform])

    testTransform = transforms.Compose([transforms.ToTensor(), normTransform])

    dataset = dset.CIFAR10
    kwargs = {'num_workers': 4, 'pin_memory': True} if useCuda else {}
    trainData = DataLoader(dataset(root='cifar', train=True, download=True, transform=trainTransform),
                           batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    testData = DataLoader(dataset(root='cifar', train=False, download=True, transform=testTransform),
                           batch_size=BATCH_SIZE, **kwargs)
    net = ResNet152()
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    loss = torch.nn.funtional.nll_loss
    date_string = datetime.datetime.now().strftime("[%m-%d-%H:%M]")
    log_dir = os.path.join('log', 'ResNet152', date_string)
    save_dir = ps.path.join('save', 'ResNet152')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_file_path = os.path.join(log_dir, 'CIFAR10_ResNet152_log.txt')
    # val_file_path = os.path.join(log_dir, 'CIFAR10_ResNet152_val.txt')
    ## write in log file
    log_file = open(log_file_path, 'w')
    # val_file = open(val_file_path, 'w')
    ## tensorboard
    summary_writer = tf.summary.FileWriter(log_dir)
    training_loss_pl = tf.placeholder(tf.float32, None)
    training_error_pl = tf.placeholder(tf.float32, None)
    val_loss_pl = tf.placeholder(tf.float32, None)
    val_error_pl = tf.placeholder(tf.float32, None)

    tf.summary.scalar('ResNet152_train_loss', training_loss_pl)
    tf.summary.scalar('ResNet152_train_error', training_error_pl)
    tf.summary.scalar('ResNet152_val_loss', val_loss_pl)
    tf.summary.scalar('ResNet152_val_error', val_error_pl)
    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto(device_count={'GPU': 0})

    ##
    train_losses = []
    train_errors = []
    val_losses = []
    val_errors = []

    best_error = 100
    best_iteration = 300
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for i in range(NUM_EPOCHS):
            sgd_lr_scheduler(optimizer, i)
            train_error, train_loss = train(net, trainData, loss, optimizer, i, NUM_EPOCHS, useCuda, log_file)
            val_error, val_loss = test(net, testData, loss, i, NUM_EPOCHS, useCuda, log_file)
            train_errors.append(train_error)
            train_losses.append(train_loss)
            val_errors.append(val_error)
            val_losses.append(val_loss)
            if val_error < best_error:
                best_iteration = i+1
                best_error = val_error
            summary_string = sess.run([summary_op], feed_dict={training_loss_pl: train_loss,
                                                               training_error_pl: train_error,
                                                               val_loss_pl: val_loss, val_error_pl: val_error})[0]
            summary_writer.add_summary(summary_string, i+1)
            ## Save the model
            torch.save(net, os.path.join(save_dir, str(int(i+1))))
            ## plot the curve
    make_plot((train_errors, train_loss, val_errors, val_losses), NUM_EPOCHS, log_dir)
    print('Finished Training. Best iteration is %d'%best_iteration)
    log_file.write('Finished Training. Best iteration is %d\n'%best_iteration)
    log_file.close()



