# Real Time Instance Segmentation using Detectron2 & Adelaidet

https://user-images.githubusercontent.com/22910010/205639811-e0631e1a-663a-4b7b-8864-2f0a8f7c5907.mp4

https://user-images.githubusercontent.com/22910010/205640944-0647d00c-38e6-4411-b7d3-047c18e83d85.mp4

https://user-images.githubusercontent.com/22910010/205890334-0cddf4a0-14ce-4269-917e-5d6885a7cd7a.mp4

## Overview:
Different Instance segmentation architectures as follows:

![insta_timeline(1)](https://user-images.githubusercontent.com/22910010/215685286-11967c60-db94-4d62-8e7b-0a7d3a650295.png)

   Here we try to solve instance segmentation on [Balloon Dataset](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip) and [Cityscapes Dataset](https://www.cityscapes-dataset.com/) using architectures,which are faster and designed to run on edge devices, such as:

  - Centermask.
  - CondInst.
  - SoloV2.
  - MaskRCNN(Not a fast architecture,but used as the benchmark for others)

## Dependency:
- [Detectron2](https://github.com/facebookresearch/detectron2): Install Detectron2 by following the instruction present.
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) : This is written on top of Detectron2.Instance Architecture such as CondInsta, Solov2,Blendmask etc. are part of it.Install AdelaiDet by following     
  instruction given in the website.Clone this repo in to the root folder of this project.

    __(Note:You may face some CUDA related errors while installing this package which you need to fix.I faced this with my RTX-2080Ti Graphics Card)__

    Please apply the patch __AdelaiDet_CUDA_fix.patch__ present in this repo.

- [CenterMask2](https://github.com/youngwanLEE/centermask2): __I modified this implementation in order to make this work in AdelaiDet with other architectures.__

  Please apply the patch __CenterMask2_modi.patch__ once you download CenterMask2.

Here are some modifications done in order to make different architectures work for instance segmentation task.

![detectron2](https://user-images.githubusercontent.com/22910010/213908903-408046a3-d8a8-4de5-9f17-a96d928d57a6.png)

## Training:

__With Balloon Dataset__:

- Download the balloon dataset and convert that in to coco format.

For training,execute the below command:

```
python3.8 training_balloon.py --arch <architectures> --path <model_out> --epochs <> model<pretrained-weight> --resume<0/1>

```

 --arch = The architecture currently supported are ['maskrcnn', 'centermask_mv2', 'centermask_v19_slimdw', 'solov2', 'condinst'].

 --path = Provide the path where model trained on Balloon dataset can be saved.

 --epochs = Provide the number of epochs.

 --model = This option is used when you want to resume the training.For example, initially you trained your model for 10000 epochs &
           the model is saved in to 'savedModels' folder.After that you to resume the training from 10000 and in that case you need to pass the model weights which was saved for
           10000 epochs inside savedModels folder.

 --resume = use 0 or 1 for resuming the training process.(Note: you have to provide the previous model weight against --model option)

For example, 
    python3.8 training_balloon.py --arch centermask_mv2 --path model_out --epochs 10000.

For resuming training, 
    python3.8 training_balloon.py --arch centermask_mv2 --path model_out --epochs 10000  --model model_out/centermask_mv2/final_model.pth --epochs 20000 --resume 1


__With Cityscapes Dataset__:

 - Register with [Cityscapes](https://www.cityscapes-dataset.com/) dataset.This might take some days.
 - Once you get the approval.Download the dataset in to your local path.
 - export DETECTRON2_DATASETS=PATH.

    For example, export DETECTRON2_DATASETS=/media/satya/work/project/segmentation/datasets.Add this to your ~/.bash.rc.

    The dataset structure should be: datasets/cityscapes/{leftImg8bit,gtFine}

 - Then run the cityscapes script as mentioned in https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html

    __(Note: I am writing an article on how to use detectron2.I will publish this soon. Please subscribe to my [Medium Blog](https://medium.com/@satya15july_11937) for the future update)__


For training, execute the below command:

```
python3.8 training_cityscapes.py --arch <architectures> --path <model_out> --epochs <> model<pretrained-weight> --resume<0/1>

```

## Inference:

__With Balloon Dataset__:

For inference, execute the below command:
```
python3.8 inference_balloon.py --arch <> --model <> --target<cpu/cuda> --source <image/webcam/video_input> --save <0/1>

```

--arch = Choose from ['maskrcnn', 'centermask_mv2', 'centermask_v19_slimdw', 'solov2', 'condinst'].

--model = Provide the path where trained model is present for the respective architecture.

--target = Choose the target device['cpu', 'cuda'].

--source= You can do the inference on image, webcam and video input.

--save = 0/1.This is valid when the inference type is 'image'.This allows you to show the segmented output on the display rather than saving the output in to "output_images" folder.

__With Cityscapes Dataset__:

For inference, execute the below command:
```
python3.8 inference_cityscapes.py --arch <> --model <> --target<cpu/cuda> --source <image/webcam/video_input> --save <0/1>

```


## Evaluation:
Here is the evaluation data measured with different architectures on CPU and GPU configuration.

![instance_segmentation_inference_time](https://user-images.githubusercontent.com/22910010/205641997-1d74d39a-5252-48d6-a63a-5004ec0f6109.png)


---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)
