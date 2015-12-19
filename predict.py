#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import argparse
import numpy as np
import cv2 as cv
import cPickle as pickle
from VGGNet import VGGNet
from chainer import cuda
from chainer import serializers
from chainer import Variable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--image', type=str, default='images/cat.jpg')
    args = parser.parse_args()

    mean = np.array([103.939, 116.779, 123.68])
    img = cv.imread(args.image).astype(np.float32)
    img -= mean
    img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    vgg = VGGNet()
    serializers.load_hdf5('VGG.model', vgg)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).user()
        vgg.to_gpu()
        img = cuda.cupy.asarray(img, dtype=np.float32)

    pred = vgg(Variable(img), None)

    if args.gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data

    words = open('data/synset_words.txt').readlines()
    words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
    words = np.asarray(words)

    top5 = np.argsort(pred)[0][::-1][:5]
    probs = np.sort(pred)[0][::-1][:5]
    for w, p in zip(words[top5], probs):
        print('{}\tprobability:{}'.format(w, p))
