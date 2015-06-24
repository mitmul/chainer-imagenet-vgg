#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cv2 as cv
import cPickle as pickle
from VGGNet import VGGNet
from chainer import cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--image', type=str, default='images/cat.jpg')
    args = parser.parse_args()

    mean = np.load('data/ilsvrc_2012_mean.npy')
    if mean.shape[1:] != (1, 1):
        mean = mean.mean(1).mean(1)

    img = cv.imread(args.image).astype(np.float32)
    img -= mean
    img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    vgg = pickle.load(open('VGGNet.chainermodel', 'rb'))
    if args.gpu >= 0:
        cuda.init()
        vgg.to_gpu()
        img = cuda.to_gpu(img.copy())

    pred = vgg.forward(img, None, train=False)

    if args.gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data

    words = open('data/synset_words.txt').readlines()
    words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
    words = np.asarray(words)

    top = words[np.argmax(pred)]
    top5 = np.argsort(pred)[0][::-1][:5]
    print 'top:', top
    print 'top5:'
    for w in words[top5]:
        print w
