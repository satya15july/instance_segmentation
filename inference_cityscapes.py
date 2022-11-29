# Usage: python3.8 inference_cityscapes.py --arch centermask_mv2 \
#                  --model model_out/centermasklite_mv2/model_final.pth --target cpu --source image --save 0

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2, random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode

from netutils import InstanceSegArch, ArchType

from utils import get_balloon_dicts
import argparse
import time
import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='mask-rcnn', choices=['maskrcnn', 'centermask_mv2', 'centermask_v19_slimdw' ,'solov2', 'condinst', 'tensormask'], help='Choose instance segmentation architecture')
ap.add_argument("-m", "--model", required=True,	help="path  of the model")
ap.add_argument('-t', '--target', default='cpu', choices=['cpu', 'cuda'], help='Choose the target device')
ap.add_argument('-s', '--source', default='image', choices=['image', 'webcam', 'video_input'], help='Choose the source type')
ap.add_argument("-f", "--file", required=False,	help="path  of the video/image file")
ap.add_argument("-s", '--save', default=0, type = int, help='save predicted output')
args = vars(ap.parse_args())

ARCHITECTURE = args['arch']
print("ARCHITECTURE: {} ".format(ARCHITECTURE))

DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_val'

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")
dataset_dicts = DatasetCatalog.get("cityscapes_fine_instance_seg_train")

model_weight = args['model']
target_device = args['target']
print("model_weight: {}, target_device: {}".format(model_weight, target_device))

arch_type = None
if args['arch'] == 'maskrcnn':
    arch_type = ArchType.MaskRCNN
elif args['arch'] == 'solov2':
    arch_type = ArchType.SoloV2
elif args['arch'] == 'condinst':
    arch_type = ArchType.CondInsta
elif args['arch'] == "centermask_mv2":
    arch_type = ArchType.CentermaskLite_MV2
elif args['arch'] == "centermask_v19_slimdw":
    arch_type = ArchType.CentermaskLite_V19_SLIM_DW
elif args['arch'] == "tensormask":
    arch_type = ArchType.TensorMask

instance_seg = InstanceSegArch(len(CLASSES), arch_type)
instance_seg.register_dataset(DATASET_TRAIN, DATASET_VAL)
instance_seg.set_model_weights(model_weight)
instance_seg.set_target_device(args['target'])
instance_seg.set_score_threshold(0.7)
instance_seg.set_confidence_threhold(0.7)

instance_seg.print_cfg()

predictor = instance_seg.default_predictor()

def run_demo_on_val_dataset():
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        print("image shape {}".format(im.shape))
        print("====PREDICTION======= STARTS")
        start = time.time()
        outputs = predictor(im)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for arch: {} on device: {} is {} ms ".format(args['arch'], target_device, elapsed_time))
        print("====PREDICTION======= ENDS")
        v = Visualizer(im[:, :, ::-1],
                       metadata=cityscapes_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(im[:, :, ::-1])  # BGR to RGB
        ax[0].set_title('Original Image ')
        ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
        ax[1].set_title('Segmented image ')
        #plt.show()
        if args['save']:
            filename = "output_images/output_{}.png".format(args['arch'])
            plt.savefig(filename, dpi=100)
        else:
            plt.show()

def filter_predictions_from_outputs(outputs,
                                    threshold=0.7,
                                    verbose=False):
    predictions = outputs["instances"].to("cpu")
    if verbose:
        print(list(predictions.get_fields()))
    indices = [i
               for (i, s) in enumerate(predictions.scores)
               if s >= threshold
               ]

    filtered_predictions = predictions[indices]

    return filtered_predictions

def run_demo_on_image():
    img_name1 = 'input_images/frankfurt_000000_003357_leftImg8bit.png'
    im = cv2.imread(img_name1)

    print("image shape {}".format(im.shape))
    print("====PREDICTION======= STARTS")
    start = time.time()
    outputs = predictor(im)
    end = time.time()
    elapsed_time = (end - start) * 1000
    print("Evaluation Time for arch: {} on device: {} is {} ms ".format(args['arch'], target_device, elapsed_time))
    #print('outputs {}'.format(outputs))
    filter_outputs=filter_predictions_from_outputs(outputs, threshold=0.5)
    #print('filter_outputs {}'.format(filter_outputs))
    print("====PREDICTION======= ENDS")
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_dicts,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    #out = v.draw_instance_predictions(filter_outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions(filter_outputs)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(im[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Segmented image ')
    # plt.show()
    if args['save']:
        filename = "output_images/output_{}.png".format(args['arch'])
        plt.savefig(filename, dpi=100)
    else:
        plt.show()

def run_demo_on_webcam(instance_seg):
    instance_seg.run_on_webcam()

def run_demo_on_video_input(instance_seg, video_input):
    output = 'video_output'
    instance_seg.run_on_video_input(video_input, output)

if args['source'] == 'image':
    #run_demo_on_image()
    run_demo_on_val_dataset()
elif args['source'] == 'webcam':
    run_demo_on_webcam(instance_seg)
elif args['source'] == 'video_input':
    video_file = args['file']
    run_demo_on_video_input(instance_seg, video_file)