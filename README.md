# chainer-imagenet-vgg

Load caffemodel and transfer it to chainermodel, then save it & use it to predict label of a image.

## Requirements

- [Chainer](http://chainer.org/) is necessary
- If you create chainermodel from caffemodel by yourself,
    - [Caffe](http://caffe.berkeleyvision.org/) is required

## Download chainermodel

If you don't want to compile caffe, there is chainermodel of VGG-16 converted from caffemodel provided [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md). It's trained on ILSVRC-2014 dataset.

To get it, just run

```
$ bash download_chainermodel.sh
```

### Run demo

```
$ python predict.py
```

It may produce the below outputs:

```
['n02124075' 'Egyptian cat']	probability:0.737295150757
['n02123045' 'tabby, tabby cat']	probability:0.17518492043
['n02123159' 'tiger cat']	probability:0.0533684939146
['n02127052' 'lynx, catamount']	probability:0.005824980326
['n04074963' 'remote control, remote']	probability:0.00200812658295
```

## Create chainermodel from caffemodel

### Download files

```
$ bash download_caffemodel.sh
```

### Load caffemodel and save it as chainermodel

```
$ python load_model.py
```
