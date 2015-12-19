#! /bin/bash
if [ ! -d data ]; then
    mkdir data
fi
cd data
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz
cd ..

if [ ! -d images ]; then
    mkdir images
fi
cd images
wget http://www.customs.go.jp/mizugiwa/maken/k9img13_golden.jpg -O dog.jpg
wget http://meoowzresq.org/wp-content/uploads/2013/01/Tabby-M.jpg -O cat.jpg
cd ..

wget https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0 -O VGG.model
