import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

# np.set_printoptions(threshold=np.nan)

rate, data = scipy.io.wavfile.read("long_test.wav")
keypress_len = 5000

# the fraction of the file we use
file_range = (0.3,0.31)

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
            data_chunks.append(current_chunk)
            current_chunk = []

    return data, is_start, data_chunks

smoothed_data, inds, chunks = \
    get_cluster_starts(data[file_range[0] * len(data): file_range[1] * len(data)])

pickle.dump(chunks, open("data.p", "wb"))

ii = np.zeros(len(smoothed_data))
ii[inds] = 1000
plt.plot(smoothed_data)
plt.plot(ii)
plt.show()
