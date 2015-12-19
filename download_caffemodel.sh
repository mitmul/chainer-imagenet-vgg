#! /bin/bash
if [ ! -d data ]; then
    mkdir data
fi
cd data
if [ ! -f VGG_ILSVRC_16_layers.caffemodel ]; then
    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi
if [ ! -f VGG_ILSVRC_16_layers_deploy.prototxt ]; then
    wget https://gist.github.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
fi
if [ ! -f synset_words.txt ]; then
    wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz
fi
cd ..

if [ ! -d images ]; then
    mkdir images
fi
cd images
wget http://www.customs.go.jp/mizugiwa/maken/k9img13_golden.jpg -O dog.jpg
wget http://meoowzresq.org/wp-content/uploads/2013/01/Tabby-M.jpg -O cat.jpg
