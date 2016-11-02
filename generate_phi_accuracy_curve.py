import numpy as np
import scipy.io
import pickle
import sklearn.linear_model
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import os

def extract_features():
  # TODO
  # Given the data sets and phi type and parameters, make the features and features file used for the below function
  pass

def make_plot_from_features(features_file, labels_file, out_file_name="phi_plot", title=None):
  # A file with comma separated values, each row contrains (absolute path to the features numpy array, phi parameter value used on the x axis)
  # The phi parameter should be a single number floating point or integer
  # file path to labels - either a numpy array or a .txt file in range [0, num classes)
  # The plot title

  assert not os.path.exists(out_file_name)
  assert not os.path.exists(out_file_name + ".png")

  # load features_file, multi thread
  lines = []
  with open(features_file) as f:
    lines = []
    for line in f:
      lines.append(line)

  print("Loading the following files: ")
  for line in lines:
    print(line)

  Xs = [np.array(1) for _ in range(len(lines))]
  phi_params = [-1 for _ in range(len(lines))]

  def load_line(index):
    line = lines[index]
    split_line = line.split(",")
    assert len(split_line == 2), "The features file should have one comma per line"
    assert os.path.exists(split_line[0]), "One of the feature paths supplied does not exists"

    X = np.load(split_line[0])

    if split_line[1][-1] == '\n':
      split_line[1] = split_line[1][:-1]
    phi_param = float(split_line[1])

    Xs[index] = X
    phi_params[index] = phi_param

  # multi thread
  pool = Pool(40)
  pool.map(load_line, range(len(lines)))

  # load labels file
  if labels_file[:-4] == ".npy"
    labels = np.load(labels_file)
  else:
    assert labels_file[:-4] == ".txt", "Should be .txt with one label per line or .npy"
    labels = []
    with open(labels_file) as f:
      for line in f:
        if line[-1] == '\n':
          line = line[:-1]
        labels.append(float(line))
    labels = np.array(labels)

  def accuracy_metric(train_X, train_y, test_X, test_y):
        start = time.time()
        print("Starting accuracy_metric calculation...")

        model = SVC(class_weight='balanced',C=0.5)
        #model = sklearn.linear_model.LogisticRegression()
        #model = GradientBoostingClassifier()

        model.fit(train_X, train_y)
        pred = model.predict(test_X)

        correct_matrix = (pred == test_y)
        accuracy = correct_matrix.sum() / correct_matrix.size

        print(accuracy, time.time()-start)
        return accuracy


  def cross_validated_accuracy_metric(X, y, num_fold=5, random_seed=0):
    kf = cross_validation.KFold(n=y.size, n_folds=num_fold, shuffle=True,
                            random_state=random_Seed)
    accuracy_metrics = [accuracy_metric(X[train_ind], y[train_ind] X[test_ind], y[test_ind]) for train_ind, test_ind in kf]
    return np.mean(accuracy_metrics)

  def produce_plot(Xs, phi_parameters, y, out_file, title=None):
    """
    y is the numpy array containing the labels 1-D tensor
    Xs is a list of numpy arrays that are 2-D tensors formated where the ith row is the set of features corresponding to the ith label in y
    phi_parameters is the list of phi parameters to be used as the value of the x axis
    """
    print("Producing plot for: ", out_file)
    print("Title is: ", title)
    assert len(Xs) == len(phi_parameters)
    points = [(-1, -1) for _ in range(len(Xs))]

    def compute_one_point(index):
      print("Computing accuracy for index: ", index, " / phi parameter: ", phi_parameters[index])
      X = Xs[index]
      accuracy_metric_score = cross_validated_accuracy_metric(X, y.copy())
      points[index] = (phi_parameters[index], accuracy_metric_score)

    pool.map(compute_one_point, range(len(Xs)))

    print("Finished assessing accuracy, now plotting")
    points = sorted(points)
    x_points = np.array([x for x, y in points])
    y_points = np.array([y for x, y in points])
    plt.plot(x_points, y_points)
    if title is not None:
      assert type(title) == type("")
      plt.title(title)
    plt.savefig(out_file)
  
  # call produce_plot
  produce_plot(Xs, phi_parameters, labels, out_file=out_file_name, title=title)


if __name__ == "__main__":
  pass
