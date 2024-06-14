from typing import Dict
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from collections import Counter
from scipy.ndimage import convolve1d
import pandas as pd
import sys
import logging
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        # kernel = gaussian(ks)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def weighted_mse_loss(df):
    #args = sys.argv[1:]
    #csv_file = args[0]
    #logging.info(f"Using FDS for [{csv_file}]!")
    #df = pd.read_csv(csv_file, header=0, usecols=[1])
    labels = df.iloc[:, 1].tolist()
    labels = np.asarray(labels)
    #_mean, _std =labels.mean(), labels.std()
    #labels = labels[(labels> _mean - 3 * _std) & (labels < _mean + 3 * _std)]
    #logging.info(f"Using FDS for [{labels.shape}]!")
    # 保存为PNG文件
    max_value = int(max(labels))+1
    min_value = int(min(labels))-1
    logging.info(f"The max value of labels is [{max(labels)}]!")
    logging.info(f"The min value of labels is [{min(labels)}]!")
    #if abs(max_value-min_value)<10:
    #    num_bins=50
    #else:
     #   num_bins=abs(max_value-min_value)
    def get_bin_idx(label, num_bins=50):
        _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=num_bins, range=(-2., 13.))
        #logging.info(f"Using FDS for [{max_value}]!")
        #logging.info(f"Using FDS for [{min_value}]!")
        #logging.info(f"Using FDS for [{num_bins}]!")
        if label > 13.:
            return num_bins - 1
        else:
            return np.where(bins_edges > label)[0][0] - 1
    bin_index_per_label = [get_bin_idx(label ,17)for label in labels]
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    #logging.info(f"Using FDS for [{len(eff_num_per_label)}]!")
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights