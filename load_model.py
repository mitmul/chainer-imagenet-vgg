#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import argparse
import chainer
import cPickle as pickle
from VGGNet import VGGNet
from chainer import cuda
from chainer import serializers
from chainer import Variable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='data/VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--prototxt', type=str,
                        default='data/VGG_ILSVRC_16_layers_deploy.prototxt')
    args = parser.parse_args()

    vgg = VGGNet()
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    for name, param in net.params.iteritems():
        layer = getattr(vgg, name)

        print name, param[0].data.shape, param[1].data.shape,
        print layer.W.data.shape, layer.b.data.shape

        layer.W = Variable(param[0].data)
        layer.b = Variable(param[1].data)
        setattr(vgg, name, layer)

    serializers.save_hdf5('VGG.model', vgg)
