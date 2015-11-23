import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA

from mlalgs import (
    get_chunk_starts,
    get_chunk_starts_simpler,
    get_features,
    clusterize,
    print_dict_sorted,
    plot_group
)

###################################################################
# Constants and debugging control
###################################################################

DATA_FILES = {
    'sound': 'sound2.wav', # sound1.wav
    'text': 'text2.txt', # text1.txt
}
# the fraction of the file we use
file_range = (0, 1.0)

USE_PCA = False
DEBUG = True
# Which plots to actually plot.
PLOT_SET = set([
    #'Segmentation Plot',
    'Cepstrum Plot',
    'PCA Plot',
])
PRINT_SET = set([
    'Segmentation',
    'Feature',
    'Clustering',
])
CURRENT_STAGE = 'Clustering' # or Segmentation, Feature, Clustering, HMM

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

rate, data = scipy.io.wavfile.read(DATA_FILES['sound'])

spaces = []
text = ''
with open(DATA_FILES['text'], "r") as f:
    text = f.read()
    for (i,c) in enumerate(text):
        if c == ' ':
            spaces.append(i)


###################################################################
# Segmentation
###################################################################

inds, ends, chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_chunk_starts_simpler(arg),
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))])

if not DEBUG:
    inds = inds.astype(int)
    ends = ends.astype(int)

if 'Segmentation' in PRINT_SET:
    print "num_chunks", len(chunks)

N = 0
M = 500000
ii = np.zeros(len(data))
ii[inds] = 1
ii2 = np.zeros(len(data))
ii2[ends] = 1

plot_group(PLOT_SET, 'Segmentation Plot',
           np.log(np.abs(data[N:M])+0.01), np.log(max(data[N:M]) * ii[N:M]+0.01), -np.log(max(data[N:M]) * ii2[N:M]+0.01))

if CURRENT_STAGE == 'Segmentation':
    sys.exit()

###################################################################
# Feature Extraction
###################################################################

features = cache_or_compute('cache/features.npy', lambda ls: map(get_features, ls), chunks)

plot_group(PLOT_SET, 'Cepstrum Plot',
            #features[10],
            #features[11],
            #features[12],
            #features[13]

           features[spaces[0]],
           features[spaces[1]],
           features[spaces[2]],
           features[spaces[3]],
           )
features = np.array(features)

if USE_PCA:
    pca = PCA(n_components=10)
    pca.fit(features)
    if 'Feature' in PRINT_SET:
        print pca.explained_variance_ratio_
    features = pca.transform(features)
    plot_group(PLOT_SET, 'PCA Plot',
               features[spaces[0]],
               features[spaces[1]],
               features[0])

if CURRENT_STAGE == 'Feature':
    sys.exit()

###################################################################
# Clustering
###################################################################

clusters, means = cache_or_compute('cache/clusters.npy', clusterize, features)
n_clusters = len(clusters)

if 'Clustering' in PRINT_SET:
    print zip(text, clusters)
    cluster_given_letter = {}
    letter_given_cluster = {}

    for char, clust in zip(text, clusters):
        if char not in cluster_given_letter:
            cluster_given_letter[char] = {}
        if clust not in letter_given_cluster:
            letter_given_cluster[clust] = {}
        cluster_given_letter[char][clust] = cluster_given_letter[char].get(clust, 0) + 1
        letter_given_cluster[clust][char] = letter_given_cluster[clust].get(char, 0) + 1

    print 'Cluster given letter'
    for char in cluster_given_letter:
        print char
        print_dict_sorted(cluster_given_letter[char])
    print 'Letter given cluster'
    for clust in letter_given_cluster:
        print clust
        print_dict_sorted(letter_given_cluster[clust])

###################################################################
# HMM Coefficient computation
###################################################################

freqs = {}
for c in clusters:
    if c not in freqs:
        freqs[c] = 0
    freqs[c] += 1

rel_freqs = {}
for c in freqs:
    rel_freqs[c] = float(freqs[c])/n_clusters

if CURRENT_STAGE == 'Clustering':
    sys.exit()

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

###################################################################
# HMM Viterbi
###################################################################

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
