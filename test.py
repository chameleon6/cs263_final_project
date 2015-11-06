import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random

def cache_or_compute(fname, fun, *args):
    try:
        return np.load(fname)
    except IOError:
        c = fun(*args)
        np.save(fname, c)
        return c

# np.set_printoptions(threshold=np.nan)

rate, data = scipy.io.wavfile.read("long_test.wav")
keypress_len = 5000

# the fraction of the file we use
file_range = (0, 1.0)

def get_cluster_starts(data):
    thresh = sorted(data)[int(0.99 * len(data))]
    is_start = data > thresh
    for (i,v) in enumerate(is_start):
        if v:
            for j in range(i+1, min(len(data), i+keypress_len)):
                is_start[j] = False

    current_chunk = []
    data_chunks = []
    for (b, d) in zip(is_start, data):
        current_chunk.append(d)
        if b:
            data_chunks.append(np.array(current_chunk))
            current_chunk = []

    return data, is_start, np.array(data_chunks)

def get_features(chunk):
    '''Grab the features from a single keystroke'''
    # The definition of cepstrum
    return (np.absolute(np.fft.ifft(np.log(np.absolute(np.fft.fft(chunk, n=500)) ** 2)))**2)[1:20]

def clusterize(ls, n=50):
    '''Clusters the objects (np arrays) in ls into n clusters
    The current algorithm is k-means
    '''

    means = random.sample(ls, n)

    def get_closest_cluster(means, l):
        smallest_i = -1
        smallest_e = 0
        for i, m in enumerate(means):
            e = np.linalg.norm(l-m)
            if e < smallest_e or smallest_i == -1:
                smallest_i = i
                smallest_e = e
        return smallest_i

    for j in range(100):
        print j
        # E-step
        z = [[] for _ in range(n)]
        for l in ls:
            z[get_closest_cluster(means, l)].append(l)

        # M-step
        for i in range(n):
            means[i] = sum(item for item in z[i])/len(z[i])

    return np.array([get_closest_cluster(means, l) for l in ls])

chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_cluster_starts(arg)[2],
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))])

#ii = np.zeros(len(smoothed_data))
#ii[inds] = 1000
#plt.plot(smoothed_data)
#plt.plot(ii)
#plt.show()

features = cache_or_compute('cache/features.npy', lambda ls: map(get_features, ls), chunks)
print features[0]
print features[1]
print features[2]

clusters = cache_or_compute('cache/clusters.npy', clusterize, features)
print clusters[:100]
