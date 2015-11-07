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
    # chunk = chunk[0:-500]
    # max_ind = np.argmax(chunk)
    # if max_ind < 0.2 * len(chunk):
    #     chunk = chunk[max_ind:]
    f=np.absolute(np.fft.fft(chunk, n=500))
    lf = np.log( f ** 2 + 0.0001)
    c = (np.absolute(np.fft.ifft(lf))**2)
    return c[1:50]

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

chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_cluster_starts(arg)[2],
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))])

chunk_lens = [len(c) for c in chunks]
inds = []
last_ind = 0
for l in chunk_lens:
    last_ind += l
    inds.append(last_ind)

"""
# segmentation plot
N = 2000000
M = 2200000
ii = np.zeros(len(data))
ii[inds] = 1000
plt.plot(data[N:M])
plt.plot(ii[N:M])
plt.show()
"""

features = cache_or_compute('cache/features.npy', lambda ls: map(get_features, ls), chunks)
print features[0]
print features[1]
print features[2]

clusters, means = cache_or_compute('cache/clusters.npy', clusterize, features)
n_clusters = len(clusters)
print clusters[:300]

freqs = {}
for c in clusters:
    if c not in freqs:
        freqs[c] = 0
    freqs[c] += 1

rel_freqs = {}
for c in freqs:
    rel_freqs[c] = float(freqs[c])/n_clusters

#plt.bar([i[0] for i in freqs.items()], [i[1] for i in freqs.items()])
#plt.show()

#cluster_dists = [[np.linalg.norm(i-j) for j in means] for i in means]
#cluster_dists_sorted = [(cluster_dists[ni][nj], ni, nj)
#        for ni in range(50) for nj in range(ni)]
#cluster_dists_sorted.sort()

cluster_dists = sorted([(sum(np.linalg.norm(i-j) for j in means), freqs[ni], ni)
    for (ni,i) in enumerate(means)])

spaces = []
with open("text.txt", "r") as f:
    s = f.read()
    for (i,c) in enumerate(s):
        if i >= len(clusters):
            break
        if c == ' ':
            spaces.append((clusters[i],i))

space_freqs = {}
for (c,ind) in spaces:
    if c not in space_freqs:
        space_freqs[c] = 0
    space_freqs[c] += 1

rel_freqs = {}
for c in space_freqs:
    rel_freqs[c] = float(space_freqs[c])/freqs[c] if freqs[c] > 0 else 0

print freqs, rel_freqs

def compute_freq_dist(data):
    counts = {}
    freq = {}
    outputs = {}

    for i, o in data:
        if i not in counts:
            counts[i] = 0
            freq[i] = {}
        if o not in freq[i]:
            freq[i][o] = 0
        freq[i][o] += 1
        counts[i] += 1
        outputs[o] = True

    for i in freq:
        for o in outputs:
            if o not in freq[i]:
                freq[i][o] = 0

    for i in freq:
        for o in freq[i]:
            freq[i][o] /= float(counts[i])

    total = 0
    for c in counts:
        total += counts[c]

    for c in counts:
        counts[c] /= float(total)

    return counts, freq

def compute_bigram():
    next_letter_prob = {}

    valid_letters = map(chr, range(97,123)) + [' ']
    with open("full_text.txt", "r") as f:
        s = f.read().lower()
        letter_pairs = np.array(zip(s, s[1:]))
        inds = np.array([True for i in range(len(letter_pairs))])
        for i, (c1,c2) in enumerate(letter_pairs):
            if c1 not in valid_letters or c2 not in valid_letters:
                inds[i] = False

        return compute_freq_dist(letter_pairs[inds])

cache_or_compute("cache/next_letter_prob.npy", compute_bigram)

pi, A = np.load("cache/next_letter_prob.npy").tolist()
valid_letters = map(chr, range(97,123)) + [' ']
n_letters = len(valid_letters)
letters_typed = [valid_letters[int(n_letters * random.random())] for i in range(n_clusters)]
eta = {}
most_likely_seq = {}
best_probs = {}
eps = 1e-7

for trial in range(100):
    print "trial", trial
    counts = {}
    eta = compute_freq_dist(zip(letters_typed, clusters))[1]

    for clust_ind, cluster in enumerate(clusters):
        best_probs_new = {}
        most_likely_seq_new = {}
        for c in A:
            eta_prob = eta[c][cluster] if c in eta else 0
            if clust_ind == 0:
                best_probs_new[c] = np.log(pi[c]) + np.log(eta_prob + eps)
                most_likely_seq_new[c] = [c]
            else:
                best_prob_new = None
                best_prob_c = None
                for old_c in A:
                    this_prob = best_probs[old_c] + np.log(A[old_c][c] + eps)
                    if best_prob_new == None or this_prob > best_prob_new:
                        best_prob_new = this_prob
                        best_prob_c = old_c

                best_probs_new[c] = best_prob_new + np.log(eta_prob + eps)

                most_likely_seq_new[c] = most_likely_seq[best_prob_c][:]
                most_likely_seq_new[c].append(c)
                #print c, most_likely_seq_new[c]

        best_probs = best_probs_new
        most_likely_seq = most_likely_seq_new

    best_prob_overall = None
    best_end = None
    for c in best_probs:
        if best_prob_overall == None or best_probs[c] > best_prob_overall:
            best_prob_overall = best_probs[c]
            best_end = c

    letters_typed = most_likely_seq[best_end]
    print "".join(letters_typed)
    assert len(letters_typed) == n_clusters
