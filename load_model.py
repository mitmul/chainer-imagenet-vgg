#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import argparse
import cPickle as pickle
from VGGNet import VGGNet
from chainer import cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--prototxt', type=str)
    args = parser.parse_args()

    vgg = VGGNet()
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    for name, param in net.params.iteritems():
        layer = getattr(vgg, name)

        print name, param[0].data.shape, param[1].data.shape,
        print layer.W.shape, layer.b.shape

        layer.W = param[0].data
        layer.b = param[1].data
        setattr(vgg, name, layer)

    pickle.dump(vgg, open('VGGNet.chainermodel', 'wb'), -1)
