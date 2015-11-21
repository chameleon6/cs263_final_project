import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
from features import mfcc
from scipy import signal

def plot_group(plot_set, title, *args):
    if title not in plot_set:
        return
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    plt.clf()
    for a, c in zip(args, colors):
        plt.plot(a, c)
    plt.title(title)
    plt.show()

def power_graph(x, fs=44100, framesz=0.01, hop=0.005):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return np.sqrt(np.sum(np.absolute(X)**2, axis=1))

preceding_gap = 1000
keypress_first_filter_len = 3000 + preceding_gap
max_keypress_len = 10000 + preceding_gap
min_keypress_len = 5000 + preceding_gap

def get_chunk_starts(data):
    '''
        Better segmentation algorithm:
            1. Segment data into small windows (very small)
            2. Calculate total power in each window
            3. Find windows exceeding threshold power.
            4. Draw another window (of size min_keypress_len) around peak, query for 25% (15%) power,
            and extract all contiguous windows exceeding this power around peak.
    '''
    power = power_graph(data)
    intervals = []
    min_size = 40
    max_size = 50
    before_peak = 6
    win = np.ones(min_size)/float(min_size)
    thresh = sorted(power)[int(0.7 * len(power))]
    peaks = np.argwhere(data > thresh)

    # For each element in peaks:
    #   draw window of size min_size, calculate sum
    #   move window, up until max_size/2.
    #   Choose window with largest sum, and center of this window is cluster center.
    #   Calculate window size: expand window size until power level reaches lower threshold, or you hit max_size
    #   append (start, end) of window to intervals.
    #   "zero" out this power.
    #   Find next unexplained peak.

    #peaks = signal.find_peaks_cwt(power, np.arange(3, 10))


    plt.plot(np.log(power))
    plt.plot(np.log(averages))
    plt.plot(np.log(np.ones(len(power))*thresh))
    #ii = np.zeros(len(power))
    #ii[peaks] = 14
    #plt.plot(ii, 'r+')
    plt.show()

    is_start = data > thresh

    starts = []

    for (i,v) in enumerate(is_start):
        if v:
            s = max(i - preceding_gap, 0)
            starts.append(s)
            for j in range(s+1, min(len(data), s+keypress_first_filter_len)):
                is_start[j] = False

    data_chunks = []
    last_max = 0
    N = len(starts)
    real_starts = []
    ends = []
    for i in range(N):
        a = starts[i]
        candidate = starts[i+1] if i < N-1 else len(data)
        b = a + max_keypress_len - preceding_gap
        current_chunk = data[a:b]
        current_max = max(np.abs(current_chunk))

        if (current_max > last_max and
                (len(data_chunks) == 0 or len(data_chunks[-1]) > min_keypress_len)) \
                or len(data_chunks[-1]) > max_keypress_len:
            data_chunks.append(np.array(current_chunk))
            real_starts.append(a)
            ends.append(b)
            last_max = current_max
        else:
            data_chunks[-1] = np.append(data_chunks[-1], np.array(current_chunk))
            ends[-1] = b
            last_max = max(current_max, last_max)

    return real_starts, ends, np.array(data_chunks)

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
            if len(z[i]) == 0:
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
