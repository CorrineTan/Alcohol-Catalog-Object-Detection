import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# import projects
ROOT_DIR = os.path.abspath("../AWS_MRCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

########## Configuration ###########

class AlcoholConfig(Config):
    NAME = "alochol"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

    # 4 class: bottle, can, carton and label. Plus background
    NUM_CLASSES = 4 + 1

########## Dataset ###########

class AlcoholDataset(utils.Dataset):
    def load_alcohol(self, dataset_dir, subset):
        self.add_class('Alcohol', 1, 'bottle')
        self.add_class('Alcohol', 2, 'carton')
        self.add_class('Alcohol', 3, 'label')
        self.add_class('Alcohol', 4, 'can')

        assert subset in ['train', 'val']
        # dataset_dir = os.path.join(dataset_dir, subset)
        annotations_path = os.path.join(dataset_dir, 'via_region_data.json')
        with open(annotations_path, 'r', encoding='utf-8') as load_f:
            strF = load_f.read()
            if len(strF) > 0:
                annotations = json.loads(strF)
            annotations = annotations['_via_img_metadata']
            annotations = list(annotations.values())
            annotations = [a for a in annotations if a['regions']]

            for a in annotations:
                # get the polygons points and class names
                polygons = [r['shape_attributes'] for r in a['regions']]
                name = [r['region_attributes']['Alcohol'] for r in a['regions']]
                alcohol_dict = {'bottle': 1, "carton": 2, 'label': 3, 'can': 4}
                alcohol_id = [alcohol_dict[a] for a in name]

                image_path = os.path.join(dataset_dir,subset, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "Alcohol",
                    image_id=a['filename'],
                    path=image_path,
                    class_id=alcohol_id,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        
        # if not alcohol dataset, delete parent class
        if image_info['source'] != 'Alcohol':
            return super(self.__class__, self).load_mask(image_id)

        # convert polygens to a bitmap mask
        alcohol_id = image_info['class_id']
        print(alcohol_id)
        # class_ids = np.array(alcohol_id, dtype=np.int32)
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            # get index of pixels inside polygens and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        class_ids = np.array(alcohol_id)
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "Alcohol":
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):

    # train datasest
    dataset_train = AlcoholDataset()
    dataset_train.load_alcohol(args.dataset, "train")
    dataset_train.prepare()

    # validate dataset
    dataset_val = AlcoholDataset()
    dataset_val.load_alcohol(args.dataset, "train")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val, 
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers='heads')


def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        print("Running on {}".format(args.image))
        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            success, image = vcapture.read()
            if success:
                image = image[..., ::-1]
                r = model.detect([image], verbose=0)[0]
                splash = color_splash(image, r['masks'])
                splash = splash[..., ::-1]
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


########## Training ###########

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AlcoholConfig()
    else:
        class InferenceConfig(AlcoholConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))






