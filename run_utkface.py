from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import math

import ctypes
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

sys.path.append('../')
import tensorflow as tf
from tensorflow.data import Iterator

from model import AlexNet, AlexNetHex

from datagenerator import ImageDataGenerator

from utils import logger_setting


# methods to prepare X_rd and X_re for NGLCM usage
def preparion(img, args):
    row = args.row
    column = args.col
    x = np.copy(img)
    x_d = np.copy(img)
    x_re = np.copy(img)

    x = x.reshape(x.shape[0], 128*128)
    x_re = x_re.reshape(x_re.shape[0], 128*128)
    x_d = x_d.reshape(x_d.shape[0], 128*128)

    direction = np.diag((-1) * np.ones(128*128))
    for i in range(128*128):
        x = int(math.floor(i / 128))
        y = int(i % 128)
        if x + row < 128 and y + column < 128:
            direction[i][i + row * 128 + column] = 1

    for i in range(x_re.shape[0]):
        x_re[i] = np.asarray(1.0 * x_re[i] * (args.ngray - 1) / x_re[i].max(), dtype=np.int16)
        x_d[i] = np.dot(x_re[i], direction)
    return x_d, x_re
# ----------------------------

def train(args):
    logger = logger_setting(f"utkface_eb{args.eb}_var{args.var}", args.ckpt_dir, True)
    num_classes = 2
    data_dir = '/data/dk/ulub/dataset/utkface'

    tr_data = ImageDataGenerator(data_dir+f'/eb{str(args.eb)}_{args.cls}_{str(args.var)}.txt',
                                 dataroot=data_dir,
                                 mode='training',
                                 batch_size=args.batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data_bias = ImageDataGenerator(data_dir+f'/eb{str(3-args.eb)}_{args.cls}_{str(args.var)}.txt',
                                  dataroot=data_dir,
                                  mode='inference',
                                  #batch_size=args.batch_size,
                                  batch_size=args.batch_size//2,
                                  num_classes=num_classes,
                                  shuffle=True)
    val_data = ImageDataGenerator(data_dir+f'/test_{args.cls}.txt',
                                  dataroot=data_dir,
                                  mode='inference',
                                  #batch_size=args.batch_size,
                                  batch_size=args.batch_size//2,
                                  num_classes=num_classes,
                                  shuffle=True)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    validation_init_op_bias = iterator.make_initializer(val_data_bias.data)

    train_batches_per_epoch = int(np.floor(tr_data.data_size / args.batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / args.batch_size))
    val_batches_per_epoch_bias = int(np.floor(val_data_bias.data_size / args.batch_size))

    x_re = tf.placeholder(tf.float32, (None, 128 * 128))
    x_d = tf.placeholder(tf.float32, (None, 128 * 128))
    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))


    model = AlexNetHex(x, y, x_re, x_d, args, Hex_flag=True)
    # model = AlexNet(x, y)

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(model.loss)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting training')
        print('load Alex net weights')
        #model.load_initial_weights(sess)

        validation = True

        val_acc = []
        val_acc_bias = []
        for epoch in range(args.epochs):

            begin = time.time()
            sess.run(training_init_op)

            train_accuracies = []
            train_losses = []
            for i in range(train_batches_per_epoch):
                batch_x, img_batch, batch_y = sess.run(next_batch)
                batch_xd, batch_re = preparion(img_batch, args)

                _, acc, loss = sess.run([optimizer, model.accuracy, model.loss], feed_dict={x: batch_x,
                                                                                            x_re: batch_re,
                                                                                            x_d: batch_xd,
                                                                                            y: batch_y,
                                                                                            model.keep_prob: 0.5,
                                                                                            model.top_k: 1})

                train_accuracies.append(acc)
                train_losses.append(loss)

                train_acc_mean = np.mean(train_accuracies[-10:])
                train_loss_mean = np.mean(train_losses[-10:])

                if (i + 1) % 10 == 0:
                    print("Epoch %d, Batch %d/%d, time = %ds, train accuracy = %.4f, loss = %.4f " % (
                        epoch, i+1, train_batches_per_epoch, time.time() - begin, train_acc_mean, train_loss_mean))

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(train_losses)

            # compute loss over validation data
            if validation:
                sess.run(validation_init_op)
                val_accuracies = []
                for i in range(val_batches_per_epoch):
                    batch_x, img_batch, batch_y = sess.run(next_batch)
                    batch_xd, batch_re = preparion(img_batch, args)
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x,
                                                              x_re: batch_re,
                                                              x_d: batch_xd,
                                                              y: batch_y,
                                                              model.keep_prob: 1.0, model.top_k: 1})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                # print("\nEpoch %d, time = %ds, validation accuracy = %.4f" % (epoch, time.time() - begik,  val_acc_mean))
                msg = "\nEpoch %d, time = %ds, validation accuracy = %.4f" % (epoch, time.time() - begin,  val_acc_mean)
                logger.info(msg)

                sess.run(validation_init_op_bias)
                val_accuracies_bias = []
                for i in range(val_batches_per_epoch_bias):
                    batch_x, img_batch, batch_y = sess.run(next_batch)
                    batch_xd, batch_re = preparion(img_batch, args)
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x,
                                                              x_re: batch_re,
                                                              x_d: batch_xd,
                                                              y: batch_y,
                                                              model.keep_prob: 1.0, model.top_k: 1})
                    val_accuracies_bias.append(acc)
                val_acc_mean_bias = np.mean(val_accuracies_bias)
                val_acc_bias.append(val_acc_mean_bias)
                # log progress to console
                # print("\nEpoch %d, time = %ds, validation accuracy = %.4f" % (epoch, time.time() - begin,  val_acc_mean_bias))
                msg = "\nEpoch %d, time = %ds, validation bias accuracy = %.4f" % (epoch, time.time() - begin,  val_acc_mean_bias)
                logger.info(msg)

            sys.stdout.flush()

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'utkface_model.ckpt')
                saver.save(sess, ckpt_file)

        ckpt_file = os.path.join(args.ckpt_dir, 'utkface_model.ckpt')
        saver.save(sess, ckpt_file)

        weights = {}
        for v in tf.trainable_variables():
            weights[v.name] = v.eval()
        np.save('./weights', weights)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    # parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size during training per GPU')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=0, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=128, help='regularization gray level')
    parser.add_argument('-eb', '--eb', type=int, default=0, help='regularization gray level')
    parser.add_argument('-var', '--var', type=int, default=0, help='regularization gray level')
    parser.add_argument('-cls', '--cls', type=str, default=0, help='regularization gray level')
    parser.add_argument('-gpu', '--gpu', type=str, default=0, help='regularization gray level')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # pretty print args
    # print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train(args)
