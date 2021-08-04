import os
import argparse
from pathlib import Path

from mob.pspm.style_network.train import train
from mob.pspm.style_network.evalution import transfer


HOME = str(Path.home())
DATASET_PATH = HOME + '/datasets/coco/raw-data/train2017'
EPOCHS = 2
BATCH_SIZE = 4

STYLE_IMAGE = './images/style/wave.jpg'
CONTENT_IMAGE = './images/content/Green_Sea_Turtle_grazing_seagrass.jpg'
RESULT_IMAGE = './images/result/result.jpg'

LEARNING_RATE = 1e1
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1e0
TV_WEIGHT = 0e0

CHECKPOINT = './model.weights'
TENSORBOARD = './tensorboard'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        metavar='<command>',
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar='DATASET_PATH',
                        help='MSCOCO TFrecord path',
                        default=DATASET_PATH)
    parser.add_argument('--style', required=False,
                        metavar='STYLE_IMAGE',
                        help='Style image to train the specific style',
                        default=STYLE_IMAGE)
    parser.add_argument('--content', required=False,
                        metavar='CONTENT_IMAGE',
                        help='Content image/video to evaluate with',
                        default=CONTENT_IMAGE)
    parser.add_argument('--batch', required=False,
                        metavar='BATCH_SIZE',
                        help='Training batch size',
                        default=BATCH_SIZE)
    parser.add_argument('--checkpoint',
                        metavar='CHECKPOINT',
                        help='dir to save checkpoint in',
                        default=CHECKPOINT)
    parser.add_argument('--tensorboard',
                        metavar='TENSORBOARD',
                        help='dir to save tensorboard log in',
                        default=TENSORBOARD)

    return parser


def check_opts(args):
    if args.command == "train":
        assert os.path.exists(args.dataset), 'dataset path not found !'
        assert os.path.exists(args.style), 'style image not found !'
        assert args.batch > 0
        assert EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert TV_WEIGHT >= 0
        assert LEARNING_RATE >= 0
    elif args.command == "evaluate":
        assert args.content, 'content image/video not found !'
        assert args.checkpoint, 'weights path not found !'


def main():
    parser = build_parser()
    args = parser.parse_args()
    check_opts(args)
    if args.command == "train":
        parameters = {
            'dataset_path' : args.dataset,
            'epochs' : EPOCHS,
            'style_image_path' : args.style,
            'content_layer_name': 'block2_conv2',
            'content_weight' : CONTENT_WEIGHT,
            'style_layer_names': [
                "block1_conv2",
                "block2_conv2",
                "block3_conv3",
                "block4_conv3"
            ],
            'style_weight' : STYLE_WEIGHT,
            'total_variation_weight' : TV_WEIGHT,
            'learning_rate' : LEARNING_RATE,
            'batch_size' : args.batch,
            'checkpoint_dir' : args.checkpoint,
            'tensorboard_dir' : args.tensorboard
        }
        train(**parameters)
    elif args.command == "evaluate":
        parameters = {
            'content' : args.content,
            'checkpoint_dir' : args.checkpoint,
            'result' : RESULT_IMAGE,
        }
        transfer(**parameters)
    else:
        print('Example usage : python main.py evaluate --content ./path/to/content/image.jpg')


if __name__ == '__main__':
    main()