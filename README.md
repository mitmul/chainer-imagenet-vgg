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
['n02123045' 'tabby, tabby cat']	probability:0.386542588472
['n02123159' 'tiger cat']	probability:0.273922026157
['n02124075' 'Egyptian cat']	probability:0.0958372727036
['n01877812' 'wallaby, brush kangaroo']	probability:0.0283357128501
['n03223299' 'doormat, welcome mat']	probability:0.0214200373739
```

## Create chainermodel from caffemodel

### Download files

```
$ bash donwload.sh
```

### Load caffemodel and save it as chainermodel

```
$ python load_model.py
```
