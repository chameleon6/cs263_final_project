import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA

from mlalgs import (
    get_chunk_starts,
    get_features,
    clusterize,
)

USE_PCA = True
DEBUG = True

def cache_or_compute(fname, fun, *args):
    if DEBUG:
        c = fun(*args)
        np.save(fname, c)
        return c
    try:
        return np.load(fname)
    except IOError:
        c = fun(*args)
        np.save(fname, c)
        return c

# np.set_printoptions(threshold=np.nan)

rate, data = scipy.io.wavfile.read("long_test_2.wav")

spaces = []
with open("text.txt", "r") as f:
    s = f.read()
    for (i,c) in enumerate(s):
        if c == ' ':
            spaces.append(i)

# the fraction of the file we use
file_range = (0, 1.0)

chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_chunk_starts(arg)[2],
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))])

chunk_lens = [len(c) for c in chunks]
chunk_maxs = [max(c) for c in chunks]
inds = []
last_ind = 0
for l in chunk_lens:
    last_ind += l
    inds.append(last_ind)

print "num_chunks", len(chunks)
# segmentation plot
N = 0
M = 500000
ii = np.zeros(len(data))
ii[inds] = 1
#ft_data, raw = stft(data[N:M])
#plt.plot(ft_data/np.max(ft_data))
#plt.plot(data[N:M:221]/float(np.max(data[N:M]))*4, "r")
plt.plot(data[N:M])
plt.plot(max(data[N:M]) * ii[N:M], "r")
plt.plot(inds[0:50], chunk_maxs[0:50])
plt.show()

features = cache_or_compute('cache/features.npy', lambda ls: map(get_features, ls), chunks)

plt.clf()
plt.plot(features[spaces[0]], 'r')
plt.plot(features[spaces[1]], 'g')
plt.plot(features[spaces[2]], 'b')
plt.plot(features[spaces[3]], 'c')
plt.plot(features[spaces[4]], 'm')
plt.show()

if USE_PCA:
    pca = PCA(n_components=20)
    pca.fit(features)
    print pca.explained_variance_ratio_
    features = pca.transform(features)

plt.clf()
plt.plot(features[spaces[0]], 'r')
plt.plot(features[spaces[1]], 'g')
plt.plot(features[spaces[2]], 'b')
plt.plot(features[spaces[3]], 'c')
plt.plot(features[spaces[4]], 'm')
plt.show()
sys.exit()

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

#sys.exit()

#plt.bar([i[0] for i in freqs.items()], [i[1] for i in freqs.items()])
#plt.show()

#cluster_dists = [[np.linalg.norm(i-j) for j in means] for i in means]
#cluster_dists_sorted = [(cluster_dists[ni][nj], ni, nj)
#        for ni in range(50) for nj in range(ni)]
#cluster_dists_sorted.sort()

cluster_dists = sorted([(sum(np.linalg.norm(i-j) for j in means), freqs[ni], ni)
    for (ni,i) in enumerate(means)])

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
letters_typed = ['t','h','e',' ']  + [valid_letters[int(n_letters * random.random())] for i in range(n_clusters-4)]
#letters_typed = [random.choice(['e','t','r','a','s',' ']) for i in range(n_clusters)]
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
