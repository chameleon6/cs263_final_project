import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
from python_speech_features.features import mfcc
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
    # Compute stft, and then adds sum of squared coeffs.
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

def get_chunk_starts_simpler(data):
    keypress_first_filter_len = 3000
    max_keypress_len = 10000
    min_keypress_len = 5000
    average_keypress_len = 7500

    thresh = sorted(data)[int(0.99 * len(data))]
    is_start = data > thresh
    starts = []
    ends = []
    for (i,v) in enumerate(is_start):
        if v:
            for j in range(i+1, min(len(data), i+keypress_first_filter_len)):
                is_start[j] = False

    current_chunk = []
    data_chunks = []
    last_max = 0
    for ind, (b, d) in enumerate(zip(is_start, data)):
        current_chunk.append(d)
        if b:
            current_max = max(current_chunk)
            if (current_max > last_max and
                    (len(data_chunks) == 0 or len(data_chunks[-1]) > min_keypress_len)) \
                    or len(data_chunks[-1]) > max_keypress_len:
                data_chunks.append(np.array(current_chunk))
            else:
                data_chunks[-1] = np.append(data_chunks[-1],np.array(current_chunk))
                #starts = starts[:-1]
            current_chunk = []
            last_max = current_max

    last_start = 0
    for i, chunk in enumerate(data_chunks):
        last_start += len(chunk)
        if last_start >= len(data) - 5:
            del data_chunks[i]
            break
        starts.append(last_start)
        ends.append(min(last_start + average_keypress_len, len(data)-1))
        data_chunks[i] = chunk[:average_keypress_len]

    #return data, is_start, np.array(data_chunks)
    return starts, ends, np.array(data_chunks)


def get_chunk_starts(data):
    '''
        Better segmentation algorithm:
            1. Segment data into small windows (very small)
            2. Calculate total power in each window
            3. Find windows exceeding threshold power.
            4. Draw another window (of size min_keypress_len) around peak, query for 25% (15%) power,
            and extract all contiguous windows exceeding this power around peak.
    '''
    hopsamp = int(0.005 * 44100)
    power = power_graph(data)
    intervals = []
    min_size = 30
    max_size = 50
    before_peak = 6
    win = np.ones(min_size)/float(min_size)
    p_sort = sorted(power)
    thresh1 = p_sort[int(0.8 * len(power))]
    thresh2 = p_sort[int(0.55* len(power))]
    thresh3 = p_sort[int(0.5 * len(power))]

    peaks = np.argwhere(power > thresh1)

    starts = []
    ends = []
    data_chunks=[]
    #plt.plot(np.log(power))
    #plt.plot(np.log(np.ones(len(power))*thresh1))
    #ii = np.zeros(len(power))
    #ii[peaks] = 14
    #plt.plot(ii, 'r+')
    #plt.show()
    for p in peaks:
        if ends and p < ends[-1]/hopsamp:
            continue
        begin = p - min_size/2 - 1
        best_center = 0
        best_size = 0
        center_val = 0
        s = np.sum(power[begin:begin+min_size])
        for center in range(p, p+max_size/2):
            size1 = min_size
            if ends:
                size2 = min(max_size, 2 * (center - ends[-1]/hopsamp))
            else:
                size2 = max_size
            for size in range(size1, size2, 2):
                begin = center - size/2
                if power[begin+size] > thresh1:
                    break
                sl = power[begin:begin+size]
                s = np.sum(sl)
                if len(np.where(sl < thresh2)) > 0.05 * size and size > size1:
                    break
                if s > center_val and begin >= p:
                    best_center = center
                    best_size = size
                    center_val = s
        if ends:
            msize = min(max_size, 2 * (center - ends[-1]/hopsamp))
        else:
            msize = max_size
        begin = best_center - best_size/2
        power[begin:begin+best_size] = thresh3
        begin = hopsamp * begin
        end = begin + hopsamp * (best_size)
        starts.append(begin)
        ends.append(end)
        data_chunks.append(data[begin:end])

    return starts, ends, np.array(data_chunks)

def get_features(chunk):
    '''Grab the features from a single keystroke'''
    # return np.float64(np.log(np.absolute(np.float32(chunk))+0.01)[:4000:4])
    # The definition of cepstrum
    f = mfcc(chunk, samplerate=44100, winlen=0.5, numcep=13)
    return f[0]
    f=np.absolute(np.fft.fft(chunk, n=256))
    lf = np.log(f ** 2)
    c = (np.absolute(np.fft.ifft(lf))**2)
    return np.log(c[0:50])

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
    l = []
    for v, k in sorted((v, k) for k, v in d.items()):
        l.append( '%s: %s' % (k, v))
    for i in range(0, len(l), 5):
        print '         '.join(l[i: i+5])
