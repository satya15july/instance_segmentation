# python3.8 training_cityscapes.py --arch centermask_mv2 --path <model_out> --epochs <> --model<> --resume<0/1>
# For example,

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2, random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from utils import get_balloon_dicts
import argparse

from netutils import InstanceSegArch, ArchType

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='maskrcnn', choices=['maskrcnn', 'centermask_mv2', 'centermask_v19_slimdw', 'solov2', 'condinst', 'tensormask'], help='Choose instance segmentation architecture')
ap.add_argument("-p", "--path", required=True,	help="output path  to the model")
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
ap.add_argument("-m", "--model", required=False,	help="Pre-trained model weight required for resume")
ap.add_argument("-r", '--resume', default=0, type=int, help='save predicted output')
args = vars(ap.parse_args())

DATASET_TRAIN = 'cityscapes_fine_instance_seg_train'
DATASET_VAL = 'cityscapes_fine_instance_seg_train'

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")
dataset_dicts = DatasetCatalog.get("cityscapes_fine_instance_seg_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cityscapes_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(img[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Segmented Image ')
    plt.show()

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

path = args['path']
model_output_path = os.path.join(path, args['arch'])
print("model_output_path {}".format(model_output_path))
os.makedirs(model_output_path, exist_ok=True)

pre_trained_weight = args['model']
if args['resume']:
    resume_flag = True
else:
    resume_flag = False

print("pre_trained_weight {} ".format(pre_trained_weight))
print("resume_flag {} ".format(resume_flag))

instance_seg = InstanceSegArch(len(CLASSES), arch_type)
instance_seg.set_model_output_path(model_output_path)
instance_seg.register_dataset(DATASET_TRAIN, DATASET_VAL)
instance_seg.print_cfg()
instance_seg.set_epochs(args['epochs'])

if resume_flag:
    instance_seg.set_model_weights(pre_trained_weight)
    instance_seg.train(resume_flag)
else:
    instance_seg.train()

