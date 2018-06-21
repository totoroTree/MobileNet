# MobileNet

A tensorflow implementation of Google's [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

The official implementation is avaliable at [tensorflow/model](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

The official implementation of object detection is now released, see [tensorflow/model/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection).

# News
YellowFin optimizer has been intergrated, but I have no gpu resources to train on imagenet with it. Call for training \~_\~

Official implement [click here](https://github.com/JianGoForIt/YellowFin)

## Base Module

<div align="center">
<img src="https://github.com/Zehaos/MobileNet/blob/master/figures/dwl_pwl.png"><br><br>
</div>

## Accuracy on ImageNet-2012 Validation Set

| Model | Width Multiplier |Preprocessing  | Accuracy-Top1|Accuracy-Top5 |
|--------|:---------:|:------:|:------:|:------:|
| MobileNet |1.0| Same as Inception | 66.51% | 87.09% |

Download the pretrained weight: [OneDrive](https://1drv.ms/u/s!AvkGtmrlCEhDhy1YqWPGTMl1ybee), [BaiduYun](https://pan.baidu.com/s/1i5xFjal) 

**Loss**
<div align="center">
<img src="https://github.com/Zehaos/MobileNet/blob/master/figures/epoch90_full_preprocess.png"><br><br>
</div>

## Time Benchmark
Environment: Ubuntu 16.04 LTS, Xeon E3-1231 v3, 4 Cores @ 3.40GHz, GTX1060.

TF 1.0.1(native pip install), TF 1.1.0(build from source, optimization flag '-mavx2')


| Device | Forward| Forward-Backward |Instruction set|Quantized|Fused-BN|Remark|
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|CPU|52ms|503ms|-|-|-|TF 1.0.1|
|CPU|44ms|177ms|-|-|On|TF 1.0.1|
|CPU|31ms| - |-|8-bits|-|TF 1.0.1|
|CPU|26ms| 75ms|AVX2|-|-|TF 1.1.0|
|CPU|128ms| - |AVX2|8-bits|-|TF 1.1.0|
|CPU|**19ms**| 89ms|AVX2|-|On|TF 1.1.0|
|GPU|3ms|16ms|-|-|-|TF 1.0.1, CUDA8.0, CUDNN5.1|
|GPU|**3ms**|15ms|-|-|On|TF 1.0.1, CUDA8.0, CUDNN5.1|
> Image Size: (224, 224, 3), Batch Size: 1

## Usage
0. Prepare Tensorflow/models/. The code for models has not been included in the tensorflow if you install tensorflow through "pip install tensorflow-gpu", so please download the code for models.

a) download models
```
sudo git clone --recurse-submodules https://github.com/tensorflow/models
```

b) complie protobuf. Run protoc at the right path, then lots of *.py files would be generated.
```
/usr/local/lib/python2.7/dist-packages/tensorflow/models/research$ sudo protoc object_detection/protos/*.proto --python_out=.
```

c) add path
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Train on Imagenet

1. Prepare imagenet data. Please refer to Google's tutorial for [training inception](https://github.com/tensorflow/models/tree/master/inception#getting-started).

2. Modify './script/train_mobilenet_on_imagenet.sh' according to your environment.

```
bash ./script/train_mobilenet_on_imagenet.sh
```

### Benchmark speed
```
python ./scripts/time_benchmark.py
```

### Train MobileNet Detector (Debugging)

1. Prepare KITTI data.

After download KITTI data (KITTI 2D object detection), you need to split it data into train/val set.
```
cd /path/to/kitti_root
mkdir ImageSets
cd ./ImageSets
ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
python ./tools/kitti_random_split_train_val.py
```

kitti_root floder then looks like below
```
kitti_root/
                  |->training/
                  |     |-> image_2/00****.png
                  |     L-> label_2/00****.txt
                  |->testing/
                  |     L-> image_2/00****.png
                  L->ImageSets/
                        |-> trainval.txt
                        |-> train.txt
                        L-> val.txt
```
Modified**
```
data/KITTI/
                  |->training/
                  |     |-> image_2/00****.png
                  |     L-> label_2/00****.txt
                  |->testing/
                  |     L-> image_2/00****.png
                  L->ImageSets/
                        |-> trainval.txt
                        |-> train.txt
                        L-> val.txt
                   L->tfrecord/
                        L->train_***.tfrecord

```

Then convert it into tfrecord. Please modify the output folder to store the tfrecord files, here for me, I've used:
--output_dir=./MobileNet/data/KITTI/tfrecord
```
python ./tools/tf_convert_data.py

```

2. Mobify './script/train_mobilenet_on_kitti.sh' according to your environment.

a. CHECK_POINT is the folder where store the pretrained weight. 

b. For the pretrained weight, please update the path to pretrained weight files inside file "checkpoint"

```
MobileNet/data/mobilenetdet-model/
	              |->checkpoint
                      |-> graph.pbtxt
                      L-> model.ckpt-906808.data-00000-of-00001
                      |-> model.ckpt-906808.index
                      L-> model.ckpt-906808.meta
```

c. update the path to pretrained weight files inside the file "checkpoint":

```
model_checkpoint_path: "/MobileNet/data/mobilenetdet-model/model.ckpt-906808"
all_model_checkpoint_paths: "/MobileNet/data/mobilenetdet-model/model.ckpt-906808"
```

3. Start training

```
bash ./script/train_mobilenetdet_on_kitti.sh
```

4. Track the training process on Tensorboard

```
tensorboard --logdir output/mobilenetdet-model/
```

5. Verification



> The code of this subject is largely based on SqueezeDet & SSD-Tensorflow.
> I would appreciated if you could feed back any bug.

## Trouble Shooting

1. About the MobileNet model size

According to the paper, MobileNet has 3.3 Million Parameters, which does not vary based on the input resolution. It means that the number of final model parameters should be larger than 3.3 Million, because of the fc layer.

When using RMSprop training strategy, the checkpoint file size should be almost 3 times as large as the model size, because of some auxiliary parameters used in RMSprop. You can use the inspect_checkpoint.py to figure it out.

2. Slim multi-gpu performance problems

[#1390](https://github.com/tensorflow/models/issues/1390)
[#1428](https://github.com/tensorflow/models/issues/1428#issuecomment-302589426)

## TODO
- [x] Train on Imagenet
- [x] Add Width Multiplier Hyperparameters
- [x] Report training result
- [ ] Intergrate into object detection task(in progress)

## Reference
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)

[SqueezeDet](https://github.com/BichenWuUCB/squeezeDet)

[Network Analysis] (https://github.com/cwlacewe/netscope)

[TF Model Usage]  https://lijiancheng0614.github.io/2017/08/22/2017_08_22_TensorFlow-Object-Detection-API/
