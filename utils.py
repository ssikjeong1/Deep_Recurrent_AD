import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sympy import Symbol, solve
import pandas as pd

sequence_length = 11

def normalize_feature(train_data):
    tmp = []
    train_feature = train_data[:, :, 8:14]
    ICV_bl = train_data[:, :, 14]
    len = np.shape(train_feature)[-1]
    mask = np.ones_like(train_feature.reshape(-1,6))
    mask[np.where(train_feature.reshape(-1, 6) == 0)] = 0
    for idx in range(len):
        data = train_feature[:,:,idx]
        norm_data = np.true_divide(data, ICV_bl)
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1, 2, 0)
    return t_tmp.astype(float), mask.reshape(-1,sequence_length ,6).astype(float)

def masking_cogntive_score(data):
    tmp = []
    max_range = [30,70,85]
    cog_feature = data.copy()
    mask = np.ones_like(cog_feature.reshape(-1,3))
    mask[np.where(cog_feature.reshape(-1,3)==0)] = 0
    for i in range(cog_feature.shape[2]):
        cog_data = cog_feature[:,:,i]
        norm_data = cog_data / max_range[i]
        tmp.append(norm_data)
        t_tmp = np.array(tmp).transpose(1,2,0)
    return t_tmp.astype(float), mask.reshape(-1, sequence_length, 3).astype(int)

def scaling_feature_t(train_feature, estim_m_out=None, estim_c_out=None, train=False):
    (b, s, f) = train_feature.shape
    tmp = train_feature.reshape(b*s, f)  # 26391 x 6
    norm_train_feature = []
    norm_estim_c = []
    norm_estim_m = []

    for idx in range(tmp.shape[1]):
        tmp_vol = tmp[:, idx]
        if train == True:
            tmp_vol_max = np.max(tmp)
            tmp_vol_min = np.min(tmp[np.nonzero(tmp)])
            m = Symbol('m')
            c = Symbol('c')
            equation1 = m * tmp_vol_max + c - 1
            equation2 = m * tmp_vol_min + c + 1
            estim_m = solve((equation1, equation2), dict=True)[0][m]
            estim_c = solve((equation1, equation2), dict=True)[0][c]
        else:
            estim_m = estim_m_out[idx]
            estim_c = estim_c_out[idx]
        norm_tmp_vol = (estim_m * tmp_vol) + estim_c
        norm_train_feature.append(norm_tmp_vol)
        norm_estim_m.append(estim_m)
        norm_estim_c.append(estim_c)
    norm_train_feature = np.array(norm_train_feature)
    norm_estim_m = np.array(norm_estim_m)
    norm_estim_c = np.array(norm_estim_c)

    norm_train_feature_t = norm_train_feature.transpose(1, 0).reshape(b, s, f)
    return norm_train_feature_t.astype(float), norm_estim_m.astype(float), norm_estim_c.astype(float)

def scaling_feature_e(train_feature, estim_m_out=None, estim_c_out=None, train=False):
    (b, s, f) = train_feature.shape
    tmp = train_feature.reshape(b*s, f)  # 26391 x 6
    norm_train_feature = []
    norm_estim_c = []
    norm_estim_m = []

    for idx in range(tmp.shape[1]):
        tmp_vol = tmp[:, idx]
        if train == True:
            tmp_vol_max = np.max(tmp[:, idx])
            tmp_vol_min = np.min(tmp[np.nonzero(tmp[:, idx]), idx])

            m = Symbol('m')
            c = Symbol('c')
            equation1 = m * tmp_vol_max + c - 1
            equation2 = m * tmp_vol_min + c + 1
            estim_m = solve((equation1, equation2), dict=True)[0][m]
            estim_c = solve((equation1, equation2), dict=True)[0][c]
        else:
            estim_m = estim_m_out[idx]
            estim_c = estim_c_out[idx]
        norm_tmp_vol = (estim_m * tmp_vol) + estim_c
        norm_train_feature.append(norm_tmp_vol)
        norm_estim_m.append(estim_m)
        norm_estim_c.append(estim_c)
    norm_train_feature = np.array(norm_train_feature)
    norm_estim_m = np.array(norm_estim_m)
    norm_estim_c = np.array(norm_estim_c)

    norm_train_feature_t = norm_train_feature.transpose(1, 0).reshape(b, s, f)
    return norm_train_feature_t.astype(float), norm_estim_m.astype(float), norm_estim_c.astype(float)


def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

import itertools
def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0].item(), instance[zero_label+1].item()))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) + a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7