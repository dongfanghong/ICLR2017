import os
import numpy as np
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow as tf

#############################################################

# Set flags
EPOCH = 25
LEARNING_RATE = 1e-4
BETA1 = 0.5
TRAIN_SIZE = np.inf
BATCH_SIZE = 36
IMAGE_SIZE = 64
DATASET = 'imagenet'
CHECKPOINT_DIR = '/atlas/u/nj/iclr2017/imagenet/downsample1_64x64/checkpoints'
SAMPLE_DIR = '/atlas/u/nj/iclr2017/imagenet/downsample1_64x64/samples'
IS_TRAIN = True
IS_CROP = True
VISUALIZE = False

#############################################################

flags = tf.app.flags
flags.DEFINE_integer("epoch", EPOCH, "Epoch to train [25]")
flags.DEFINE_float(
    "learning_rate", LEARNING_RATE, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", BETA1, "Momentum term of adam [0.5]")
flags.DEFINE_integer(
    "train_size", TRAIN_SIZE, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", BATCH_SIZE, "The size of batch images [64]")
flags.DEFINE_integer(
    "image_size", IMAGE_SIZE,
    "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string(
    "dataset", DATASET, "The name of dataset [imagenet, celebA, mnist, lsun]")
flags.DEFINE_string(
    "checkpoint_dir", CHECKPOINT_DIR,
    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string(
    "sample_dir", SAMPLE_DIR,
    "Directory name to save the image samples [samples]")
flags.DEFINE_boolean(
    "is_train", IS_TRAIN, "True for training, False for testing [False]")
flags.DEFINE_boolean(
    "is_crop", IS_CROP, "True for training, False for testing [False]")
flags.DEFINE_boolean(
    "visualize", VISUALIZE, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

#############################################################


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    # Make sure the checkpoint and sample directories exist
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    # Start training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                y_dim=10, dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir)
        elif FLAGS.dataset == 'imagenet':
            dcgan = DCGAN(
                sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir)
        else:
            dcgan = DCGAN(
                sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir)
        # Determine training or test
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
        # Visualize stuff
        if FLAGS.visualize:
            to_json(
                "./web/js/layers.js",
                [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                [dcgan.h4_w, dcgan.h4_b, None])
            # Codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
