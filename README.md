# ViSCAN
This is the implementation of our paper：**From Still to Moving: Enhancing Real-Time Metal Corrosion Detection with Dual Feature Aggregation**

Link subsequent updates.

## Dataset

For our TJNC dataset see [TJNC Dataset]()

## Pretrain Models

Pretrained Models for YOLOV: [YOLOV](https://drive.google.com/file/d/1URbEAVIQH1azU1kmXfCwvpciHjTseA6c/view?usp=drive_link)

Pretrained Models for YOLOX: [YOLOX](https://drive.google.com/file/d/1F2-zF9j0ihX6K3GZUiKRG2ZuiYqvXy-7/view?usp=drive_link)

## Quick Start

### Installation for Enviroment

```
git clone https://github.com/returnlsz/ViSCAN.git

cd ViSCAN

conda create -n yolov python=3.7

conda activate yolov

pip install -r requirements.txt

pip3 install -v -e .
```

### Data Preparation

#### Data Structure

We use the data set format of `ILSVRC2015`, as follows:

```
|-ILSVRC2015
|----Annotations
|----|----VID
|----|----|----train
|----|----|----|----ILSVRC2015_VID_train_0000
|----|----|----|----|----ILSVRC2015_train_00000000
|----|----|----|----|----|----00000.xml
|----|----|----|----|----|----00001.xml
|----|----|----|----|----|----...
|----|----|----|----|----ILSVRC2015_train_00000001
|----|----|----|----|----...
|----|----|----|----ILSVRC2015_VID_train_0000_augmented
|----|----|----val
|----|----|----|----ILSVRC2015_val_00003003
|----|----|----|----...
|----|----|----val_augmented
|----Data
|----|----VID
|----|----|----train
|----|----|----|----ILSVRC2015_VID_train_0000
|----|----|----|----|----ILSVRC2015_train_00000000
|----|----|----|----|----|----00000.jpg
|----|----|----|----|----|----00001.jpg
|----|----|----|----|----|----...
|----|----|----|----|----ILSVRC2015_train_00000001
|----|----|----|----|----...
|----|----|----|----ILSVRC2015_VID_train_0000_augmented
|----|----|----val
|----|----|----|----ILSVRC2015_val_00003003
|----|----|----|----...
|----|----|----val_augmented
```

#### For ViSCAN training

For the training of ViSCAN, we use the TJNC video dataset, and after preparing the framework behind our TJNC dataset, use the following instructions

```
python generate_npy.py
```

This will generate `train_seq.npy` and `val_seq.npy` files for ViSCAN training and testing (files with the npy suffix cannot be opened directly, you can use the `view_npy.py` script to convert the npy file to a txt file to see its details).

**You need to review the instructions in the `generate_npy.py` script to make changes to the corresponding path** .

#### For YOLOX training

We sample the TJNC video data set to get the corresponding image data set, which will be used for YOLOX training, after preparing our `train_seq.npy` and `val_seq.npy` files, using the following instructions

```
python xml2coco.py
```

This will be used for the `vid_train_coco.json` and `vid_val_coco.json` files for YOLOX training and testing (default under annotations folder).

**You need to modify the values of `data_dir` and `jsonname` at the bottom of the script `xml2coco.py`.**

### Training Pipelines

Our ViSCAN training is divided into two processes:

1. **Use the pre-trained yoloxl_vid to train on the image dataset**, using the following instructions

```
python tools/train.py -f exps/yolov/yoloxl_vid.py -b 16 --fp16 -o -c weights/yoloxl_vid.pth
```

Among them, `weights/yoloxl_vid.pth` is the YOLOX pre-training weights provided by us, and `exps/yolov/yoloxl_vid.py` is the training configuration file where you can modify the training parameters (we recommend the default parameters). You can view other optional configurations for training at `train.py`.

2. **Use YOLOX_L trained in 1 to train on our TJNC video dataset** (training on TJNC dataset directly using YOLOV's pre-training weight cannot achieve the expected effect)

Use the following instructions

```
python tools/vid_train.py -f exps/yolov/yolov_l.py -c weights/yolov_l.pth --fp16 -b 32
```

Among them, `weights/yolov_l.pth` is the trained YOLOX model obtained in step 1, and we will train ViSCAN on this basis. `exps/yolov/yolov_l.py` is the training configuration file.

### Evalution

Use the following instructions for evaluation

```
python tools/vid_eval.py -f exps/yolov/yolov_l.py --mode 'lc' -d 0 --lframe 32 --gframe 0 -c path_to_your_weights.pth --tnum 500 --fp16
```

Where, `mode` is the frame sampling strategy, which you can see in our paper, `lc` is the proximity sampling, `lframe` and `gframe` are the number of frames sampled, more optional parameters see `vid_eval.py`.

### Inference

Use the following instructions to inference

```
python tools/vid_demo.py -f exps/yolov/yolov_l.py -c path_to_your_weights --path path_to_your_video.mp4 --conf 0.25 --nms 0.5 --tsize 576 --save_result true --device gpu
```

Where `path_to_your_video.mp4` is your audio file, see `vid_demo.py` for more inference options.

## Cite ViSCAN
