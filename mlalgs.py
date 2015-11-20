import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
from features import mfcc

def plot_group(plot_set, title, *args):
    if title not in plot_set:
        return
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    plt.clf()
    for a, c in zip(args, colors):
        plt.plot(a, c)
    plt.title(title)
    plt.show()

def stft(x, fs=44100, framesz=0.01, hop=0.005):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return np.sqrt(np.sum(np.absolute(X)**2, axis=1)), X

keypress_first_filter_len = 3000
max_keypress_len = 10000
min_keypress_len = 5000

def get_chunk_starts(data):
    thresh = sorted(data)[int(0.99 * len(data))]
    is_start = data > thresh
    for (i,v) in enumerate(is_start):
        if v:
            for j in range(i+1, min(len(data), i+keypress_first_filter_len)):
                is_start[j] = False

    current_chunk = []
    data_chunks = []
    last_max = 0
    for (b, d) in zip(is_start, data):
        current_chunk.append(d)
        if b:
            current_max = max(current_chunk)
            if (current_max > last_max and
                    (len(data_chunks) == 0 or len(data_chunks[-1]) > min_keypress_len)) \
                    or len(data_chunks[-1]) > max_keypress_len:
                data_chunks.append(np.array(current_chunk))
            else:
                data_chunks[-1] = np.append(data_chunks[-1],np.array(current_chunk))
            current_chunk = []
            last_max = current_max

    return data, is_start, np.array(data_chunks)

def get_features(chunk):
    '''Grab the features from a single keystroke'''
    # The definition of cepstrum
    f = mfcc(chunk)
    return np.concatenate((f[2], f[3]))
    chunk = chunk[0:-500]
    max_ind = np.argmax(chunk)
    if max_ind < 0.2 * len(chunk):
        chunk = chunk[max_ind:]
    f=np.absolute(np.fft.fft(chunk, n=500))
    lf = np.log(f ** 2 + 0.0001)
    c = (np.absolute(np.fft.ifft(lf))**2)
    return c[5:len(c)/4]

def clusterize(ls, n=50):
    '''Clusters the objects (np arrays) in ls into n clusters
    The current algorithm is k-means
    '''

    means = random.sample(ls, n)
    assignments = [0 for i in range(len(ls))]
    dead_cluster = {}

    def get_closest_cluster(means, l):
        smallest_i = -1
        smallest_e = 0
        for i, m in enumerate(means):
            if i in dead_cluster:
                continue
            e = np.linalg.norm(l-m)
            if e < smallest_e or smallest_i == -1:
                smallest_i = i
                smallest_e = e
        return smallest_i

    for j in range(40):

        changes = 0
        # E-step
        z = [[] for _ in range(n)]
        for (ind, l) in enumerate(ls):
            new_cluster = get_closest_cluster(means, l)
            z[new_cluster].append(l)
            if new_cluster != assignments[ind]:
                changes += 1
            assignments[ind] = new_cluster

        # M-step
        for i in range(n):
            if len(z[i]) = 0:
                dead_cluster[i] = True

            if i in dead_cluster:
                continue
            means[i] = sum(item for item in z[i])/len(z[i])

        print "iteration:", j, "num_changes:", changes
        if changes < 5:
            break

    return np.array([get_closest_cluster(means, l) for l in ls]), means


def print_dict_sorted(d):
    for v, k in sorted((v, k) for k, v in d.items()):
        print '%s: %s' % (k, v)
