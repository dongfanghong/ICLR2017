import numpy as np
import scipy.io
import pickle
import sklearn.linear_model
from sklearn import cross_validation
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import os
import argparse

from utils import find_files

TEST_DATA_DIR = '/atlas/u/nj/imagenet/ILSVRC2012_img_val_64x64/'
TEST_DATA_LABELS = '/atlas/u/nj/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

DOG_TEST_DATA_LABELS = '/atlas/u/nj/imagenet/ILSVRC2012_devkit_t3/data/ILSVRC2012_validation_ground_truth.txt'
NOT_A_DOG_LABEL_VALUE = -1

# Set flags - MAKE SURE THESE MATCH THE PARAMETERS USED TO CREATE THE MODEL
# Flags commented out do not matter
GPU_FRACTION = 0.15 # was 0.25 during training but lowering so 6 can be run on one GPU, should work since we don't use the generator at all
#EPOCH = 25
#LEARNING_RATE = 1e-4
#BETA1 = 0.5
#TRAIN_SIZE = np.inf
BATCH_SIZE = 36
IMAGE_SIZE = 64
DATASET = 'imagenet'
#CHECKPOINT_DIR = '/atlas/u/nj/iclr2017/imagenet/test/checkpoints' # This does matter but is passed in as a parameter
#SAMPLE_DIR = '/atlas/u/nj/iclr2017/imagenet/test/samples'
#EVAL_DIR = '/atlas/u/nj/iclr2017/imagenet/test/eval'
#LOG_DIR = '/atlas/u/nj/iclr2017/imagenet/test/logs'
# IS_TRAIN = True
IS_CROP = True
# VISUALIZE = False

EXTRACT_FEATURES_POOL_SIZE = int(np.floor(1.0/GPU_FRACTION))
extract_feature_lines = None
extract_features_out_names = None

def single_extract_feature(checkpoint_dir, phi_param_val, out_name=None):
  print("loading tensorflow and extracting features for: ", checkpoint_dir)
  import tensorflow as tf
  from model import DCGAN
  print("Done importing tensorflow for: ", checkpoint_dir)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRACTION)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    dcgan = DCGAN(
          sess, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
          dataset_name=DATASET, is_crop=IS_CROP,
          checkpoint_dir=checkpoint_dir)
    dcgan.load(checkpoint_dir)

    # read all the images from /atlas/u/nj/imagenet/ILSVRC2012_img_val_64x64 in the same order as they are in labels and extract
    # The features, make sure they get saved in the same order as labels is in
    def extract_image_number(image_file_name):
      # exmaple name: ILSVRC2012_val_00000001.JPEG
      number_and_extension = image_file_name.split('_')[-1]
      number = number_and_extension.split('.')[0]
      return int(number)

    # TODO do this once somewhere else instead of multiple times here
    file_names_list = find_files(TEST_DATA_DIR, '*.JPEG') # glob should work if this does not
    print("Number of file names found when extracting features: ", len(file_names_list))
    file_names_list = [(extract_image_number(name), name) for name in file_names_list]
    file_names_list.sort()
    file_names_list = [name for number, name in file_names_list]

    if out_name == None:
       out_name = str(phi_param_val) + '_features.npy',

    print("Starting extraction for: ", checkpoint_dir, phi_param_val, out_name)
    dcgan.extract_features(file_names_list, out_name, IMAGE_SIZE, IS_CROP)

def parse_line(line):
   split_line = line.split(',')
   assert (len(split_line) == 2), "The extract_features file should have one comma per line"
   assert os.path.exists(split_line[0]), "One of the checkpoint paths supplied does not exists"

   checkpoint_dir = split_line[0]

   if split_line[1][-1] == '\n':
     split_line[1] = split_line[1][:-1]
   phi_param = float(split_line[1])
   return checkpoint_dir, phi_param

def single_extract_feature_and_load_line(index):
   line = extract_feature_lines[index]

   checkpoint_dir, phi_param = parse_line(line)

   out_name = extract_features_out_names[index]
   print("Extract Features: parsed phi ", phi_param)
   if not os.path.exists(out_name):
      single_extract_feature(checkpoint_dir, phi_param, out_name)
   else:
      print("Skipping: ", out_name)
   

def extract_features(extract_features_config_file, run_name):

  print("Extracting features using pool size: ", EXTRACT_FEATURES_POOL_SIZE)
  global extract_feature_lines, extract_features_out_names

  extract_feature_lines = []
  extract_features_out_names = []
  phi_params = []
  with open(extract_features_config_file) as f:
     for line in f:
        extract_feature_lines.append(line)
        checkpoint_dir, phi_param = parse_line(line)
        phi_params.append(phi_param)
        extract_features_out_names.append(run_name + "_" + str(phi_param) + "_features.npy")

  print("Read extract features input file")
  print(extract_features_config_file)

  # Given the data sets and phi type and parameters, make the features and features file used for the below function
  features_config_file_name = run_name + "_plot_features_config_file.txt"

  features_config_file_content = ""
  for index, feature_array_name in enumerate(extract_features_out_names):
     phi_val = phi_params[index]
     features_config_file_content += feature_array_name + "," + str(phi_val) + '\n'

  print("Writing config file for the classification phase after features are extracted")

  with open(features_config_file_name, 'w') as f:
     f.write(features_config_file_content)

  print("Creating process pool for the following lines:")
  print(extract_feature_lines)
  print(extract_features_out_names)

  pool = Pool(EXTRACT_FEATURES_POOL_SIZE)
  pool.map(single_extract_feature_and_load_line, range(len(extract_feature_lines)))

  print("Finished extracting features and returning features_config_file_name")

  return features_config_file_name

#############################################################################################
################################# PLOTTING CODE SECTION #####################################
#############################################################################################

"""
These variables and functions must be defined at the top level to work with the pool
Every variable and function below this point belongs to make_plot_from_features
"""
lines = None
labels = None

def accuracy_metric_impl(train_X, train_y, test_X, test_y, model, name=None):
        start = time.time()
        print(name, "Starting accuracy_metric calculation...")
        print(name, "With top 5 accuracy, 1 fold, X train shape: ", train_X.shape, " X test shape: ", test_X.shape)

        model.fit(train_X, train_y)

        #pred = model.predict(test_X)

        #correct_matrix = (pred == test_y)
        #accuracy = float(correct_matrix.sum()) / float(correct_matrix.size)

        #print("Accuracy: ", accuracy, " Time: ", time.time()-start)
        #return accuracy

        print(name, "Model classes shape: ", model.classes_.shape)
        log_prob = model.predict_log_proba(test_X)
        order = np.argsort(log_prob, axis=1)
        top_5 = order[:, -5:]
        total = 0
        in_top_5 = 0
        for row_index in range(test_y.shape[0]):
            total += 1
            top_5_classes = model.classes_[top_5[row_index, :]]
            if test_y[row_index] in top_5_classes:
               in_top_5 += 1
        accuracy = float(in_top_5)/float(total)
        print(name, "Top 5 accuracy: ", accuracy, " Time: ", time.time()-start)
        return accuracy


def accuracy_metric(train_X, train_y, test_X, test_y):
        model = SVC(class_weight='balanced', C=0.5, kernel='linear', decision_function_shape='ovr', probability=True)
        SVM_accuracy = accuracy_metric_impl(train_X, train_y, test_X, test_y, model, name="SVC_linear")

        model = sklearn.linear_model.LogisticRegression()
        logistic_accuracy = accuracy_metric_impl(train_X, train_y, test_X, test_y, model, name="logistic_regression")

        model = GradientBoostingClassifier()
        gradient_accuracy = accuracy_metric_impl(train_X, train_y, test_X, test_y, model, name="Gradient boosting classifier")

        return SVM_accuracy, logistic_accuracy, gradient_accuracy

def cross_validated_accuracy_metric(X, y, num_fold=5, random_seed=0):
    actual_examples = (y != NOT_A_DOG_LABEL_VALUE)
    X = X[actual_examples, :]
    y = y[actual_examples]

    kf = model_selection.StratifiedKFold(n_splits=num_fold, shuffle=True,
                            random_state=random_seed)
    #accuracy_metrics = [list(accuracy_metric(X[train_ind], y[train_ind], X[test_ind], y[test_ind])) for train_ind, test_ind in kf.split(X, y)]
    # for speed purposes, just try one fold
    for train_ind, test_ind in kf.split(X, y):
       return accuracy_metric(X[train_ind], y[train_ind], X[test_ind], y[test_ind])
    return np.mean(np.array(accuracy_metrics), axis=1)

def compute_one_point(X, labels, phi_param):
      print(X.shape, labels.shape, phi_param)
      print("Computing accuracy for phi parameter: ", phi_param)
      accuracy_metric_score = cross_validated_accuracy_metric(X, labels)
      return (phi_param, accuracy_metric_score[0], accuracy_metric_score[1], accuracy_metric_score[2])

def load_line(index):
    line = lines[index]
    print("Loading line: ", line)
    split_line = line.split(",")
    assert (len(split_line) == 2), "The features file should have one comma per line"
    assert os.path.exists(split_line[0]), "One of the feature paths supplied does not exists"

    X = np.load(split_line[0])

    if split_line[1][-1] == '\n':
      split_line[1] = split_line[1][:-1]
    phi_param = float(split_line[1])

    print("Loaded phi: ", phi_param)
    return compute_one_point(X, labels, phi_param)


def make_plot_from_features(features_file, labels_file, out_file_name="phi_plot", title=None):
  # A file with comma separated values, each row contrains (absolute path to the features numpy array, phi parameter value used on the x axis)
  # The phi parameter should be a single number floating point or integer
  # file path to labels - either a numpy array or a .txt file in range [0, num classes)
  # The plot title

  global lines, Xs, phi_params, labels

  assert not os.path.exists(out_file_name)
  assert not os.path.exists(out_file_name + ".png")

  # load features_file, multi thread
  print("Reading features file: ", features_file)
  lines = []
  with open(features_file) as f:
    for line in f:
      lines.append(line)

  print("Loading the following files: ")
  for line in lines:
    print(line)

  # load labels file
  print("Reading labels file: ", labels_file)
  print("File extension: ", labels_file[-4:])
  if labels_file[-4:] == ".npy":
    labels = np.load(labels_file)
  else:
    assert labels_file[-4:] == ".txt", "Should be .txt with one label per line or .npy"
    labels = []
    with open(labels_file) as f:
      for line in f:
        if line[-1] == '\n':
          line = line[:-1]
        labels.append(float(line))
    labels = np.array(labels)

  # multi thread
  pool = Pool(40)
  points = pool.map(load_line, range(len(lines)))

  # Produce the plot
  print("Producing plot for: ", out_file_name)
  print("Title is: ", title)

  points = sorted(points)
  cache_points = np.zeros((len(points), 4))
  for i, xy in enumerate(points):
     for j in range(4):
         cache_points[i, j] = xy[j]

  np.save(out_file_name + "_points.npy", cache_points)

  print("Points:")
  print(points)
  x_points = np.array([x for x, y_SVM, y_logistic, y_boosting in points])
  svm_y_points = np.array([y_SVM for x, y_SVM, y_logistic, y_boosting in points])
  logistic_y_points = np.array([y_logistic for x, y_SVM, y_logistic, y_boosting in points])
  boosting_y_points = np.array([y_boosting for x, y_SVM, y_logistic, y_boosting in points])
  plt.plot(x_points, svm_y_points, 'r', label="SVM")
  plt.plot(x_points, logistic_y_points, 'b', label="Logistic")
  plt.plot(x_points, boosting_y_points, 'g', label="Boosting")
  plt.legend(loc='best')
  if title is not None:
    assert type(title) == type("")
    plt.title(title)
  plt.savefig(out_file_name)
  
def extract_and_plot(extract_features_config_file, run_name, labels_file, plot_out_file, plot_title):
    plot_config_file = extract_features(extract_features_config_file, run_name)
    make_plot_from_features(plot_config_file, labels_file, plot_out_file, plot_title)

if __name__ == "__main__":
    out_file_default_placeholder = "This is the default and will be changed, you did not set this param"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'run_name', help="Please give different experiments different names")
    parser.add_argument(
        'config_file', help='A file where each line is "[1],[2]\n" where [1] is the checkpoint directory and [2] is the phi value')
    parser.add_argument(
        '--out_file', help='The plot file name', default=out_file_default_placeholder)
    parser.add_argument(
        '--title', help="The title of the plot", default=None)
    parser.add_argument(
        '--data_dir', help='The directory with images in it, currently the code expects this to be imagenet format where images are X_<number>.JPEG', default=TEST_DATA_DIR)
    parser.add_argument(
        '--labels_file', help='A numpy array of labels or txt file of labels with one per line, should correspond to the features in each features array', default=TEST_DATA_LABELS)
    args = parser.parse_args()

    out_file = args.out_file
    if args.out_file == out_file_default_placeholder:
       out_file = args.run_name + "_plot"

    print("Args:")
    print(args)

    global TEST_DATA_DIR
    TEST_DATA_DIR = args.data_dir

    extract_and_plot(args.config_file, args.run_name, args.labels_file, out_file, args.title)










