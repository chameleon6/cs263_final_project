import sys
import math
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
import re

from mlalgs import (
    load_data,
    get_chunk_starts,
    get_chunk_starts_simpler,
    clusterize,
    print_dict_sorted,
    plot_group,
    baum_welch,
    longest_common_subseq,
    spell_check
)

from features import (
    get_features, get_features_fft, get_features_cepstrum
)

from langmodel import (
    valid_letters,
    compute_freq_dist,
    compute_bigram
)

###################################################################
# Constants and debugging control
###################################################################

DATA_FILES = {
    'sound': 'data/sound7.wav', # sound1.wav
    'text': 'data/text7.txt', # text1.txt
}
# the fraction of the file we use
file_range = (0, 1.0)

SOFT_CLUSTER = True
USE_PCA = True
DEBUG = False
# Which plots to actually plot.
PLOT_SET = set([
    #'Segmentation Plot',
    #'Cepstrum Plot',
    #'PCA Plot',
])
PRINT_SET = set([
    'Segmentation',
    'Feature',
    #'Clustering',
])
CURRENT_STAGE = 'HMM' # or Segmentation, Feature, Clustering, HMM

def cache_or_compute(fname, fun, *args, **kwargs):
    if DEBUG or ("debug" in kwargs and kwargs["debug"]):
        c = fun(*args)
        np.save(fname, c)
        return c
    try:
        print "Trying to use cache at", fname, "for", fun
        return np.load(fname)
    except IOError:
        print "cache not found. Recomputing"
        c = fun(*args)
        np.save(fname, c)
        return c

rate, data, text = load_data(DATA_FILES['sound'], DATA_FILES['text'])

pi, A, pi_v, theta_v, word_freq = cache_or_compute("cache/bigram.npy", compute_bigram,
        debug=False)

spaces = []
for (i,c) in enumerate(text):
    if c == ' ':
        spaces.append(i)

###################################################################
# Segmentation
###################################################################

inds, ends, chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_chunk_starts(arg),
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))], debug=False)

if not isinstance(inds, list):
    inds = inds.astype(int)
    ends = ends.astype(int)

if 'Segmentation' in PRINT_SET:
    print "num_chunks", len(chunks)

'''
TODO: Let's move this stuff out of this file? To supervised or a new file, so that this file contains only the HMM-related stuff

chunk_lens = map(lambda (x,y): x-y, zip(np.append(inds[1:], [len(data)]), inds))
spacinesses = np.array([x+y for (x,y) in zip([0] + chunk_lens, chunk_lens)])
space_thresh = sorted(spacinesses)[int(0.5 * len(spacinesses))]
space_guesses_from_time = []

for i,v in enumerate(spacinesses):
    if v > space_thresh:
        should_append = True
        if len(space_guesses_from_time) > 0 and space_guesses_from_time[-1] == i-1:
            if v < spacinesses[i-1]:
                should_append = False
            else:
                space_guesses_from_time = space_guesses_from_time[:-1]
        if should_append:
            space_guesses_from_time.append(i)

#N, M = len(data) - 500000, len(data)
N, M = 0, 1000000
ii = np.zeros(len(data))
ii[inds] = 1
ii2 = np.zeros(len(data))
ii2[ends] = 1

plot_ind_start = None
plot_ind_end = None
for i,v in enumerate(inds):
    if v > N and plot_ind_start == None:
        plot_ind_start = i
    if v > M:
        plot_ind_end = i+1
        break

ii_space = np.zeros(len(data))

typed_words = text.split(' ')
typed_lens = map(len, typed_words)
guessed_lens = [x-y-1 for (x,y) in zip(space_guesses_from_time, [-1] + space_guesses_from_time)]
typed_lens = np.array(typed_lens)
guessed_lens = np.array(guessed_lens)
print "actual word lens:"
print typed_lens
print "guessed lens:"
print guessed_lens

typed_word_inds, guessed_word_inds, good_word_lens = longest_common_subseq(typed_lens, guessed_lens)
starts = np.array(inds)[space_guesses_from_time]
supervised_train_data = []

for t,g,l in zip(typed_word_inds, guessed_word_inds, good_word_lens):
    real_word = typed_words[t]
    this_chunk = space_guesses_from_time[g-1]+1 if g > 0 else 0
    print real_word, this_chunk
    for i in range(l):
        supervised_train_data.append((real_word[i], chunks[this_chunk + i]))

for i in starts:
    ii_space[i] = 1

plot_group(PLOT_SET, 'Segmentation Plot',
           np.log(np.abs(data[N:M]+0.01)),
           np.log(max(data[N:M]) * ii_space[N:M]+0.01),
           -np.log(max(data[N:M]) * ii2[N:M]+0.01),
           #(np.array(inds[plot_ind_start:plot_ind_end]) - N,
           #    [np.log(max(chunk)+0.01) for chunk in chunks[plot_ind_start:plot_ind_end]]),
           (np.array(inds[plot_ind_start:plot_ind_end]) - N,
               spacinesses[plot_ind_start:plot_ind_end]/1600),
           np.ones(M-N)*space_thresh/1600
           )

'''
if CURRENT_STAGE == 'Segmentation':
    sys.exit()

###################################################################
# Feature Extraction
###################################################################

features = cache_or_compute('cache/features.npy', get_features, data, inds, ends, debug=False)

plot_group(PLOT_SET, 'Cepstrum Plot',
            #features[10],
            #features[11],
            #features[12],
            #features[13]
           features[spaces[0]],
           features[spaces[1]], features[spaces[2]],
           features[0], features[1])
features = np.array(features)

if USE_PCA:
    pca = PCA(n_components=80)
    pca.fit(features)
    if 'Feature' in PRINT_SET:
        print "explained variances", pca.explained_variance_ratio_
        print "total explained var", sum(pca.explained_variance_ratio_)

    features = pca.transform(features)
    plot_group(PLOT_SET, 'PCA Plot',
               features[spaces[0]],
               features[spaces[1]],
               features[spaces[2]],
               features[0], features[1])

if CURRENT_STAGE == 'Feature':
    sys.exit()

'''
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=27, covariance_type="diag", n_iter=1000).fit(features)
hidden_states = model.predict(feature_v)
print hidden_states

sys.exit()
'''

###################################################################
# Clustering
###################################################################

for _ in range(1):
    print "beginning clustering"
    clusters, means = cache_or_compute('cache/clusters.npy', clusterize, features, spaces,
            SOFT_CLUSTER, debug=False)
    n_clusters = len(clusters)

    if 'Clustering' in PRINT_SET:
        print zip(text, clusters)
        cluster_given_letter = {}
        letter_given_cluster = {}

        log_likeliness = 0
        for char, clust in zip(text, clusters):
            if char not in cluster_given_letter:
                cluster_given_letter[char] = {}
            if clust not in letter_given_cluster:
                letter_given_cluster[clust] = {}
            cluster_given_letter[char][clust] = cluster_given_letter[char].get(clust, 0) + 1
            letter_given_cluster[clust][char] = letter_given_cluster[clust].get(char, 0) + 1
        for char, clust in zip(text, clusters)[:60]: # not sure when we make first chunk mistake
            s = sum(letter_given_cluster[clust].values())
            log_likeliness += math.log(letter_given_cluster[clust][char]/float(s))

        print 'Cluster given letter'
        for char in cluster_given_letter:
            print "CHARACTER:", char
            print_dict_sorted(cluster_given_letter[char])
        print 'Letter given cluster'
        for clust in letter_given_cluster:
            print "CLUSTER:", clust
            print_dict_sorted(letter_given_cluster[clust])

        print 'Log likelihood given cluster: ', log_likeliness

    if CURRENT_STAGE == 'Clustering':
        sys.exit()

    ###################################################################
    # HMM Coefficient computation
    ###################################################################

    spaces_bw = [c for s, c in zip(text, clusters) if s == ' ']
    phi, score, seq, gamma = cache_or_compute("hmm.npy", baum_welch, pi_v,
            theta_v, clusters, spaces_bw, text, SOFT_CLUSTER, debug=False)
    spell_checked_seq = spell_check(gamma, seq, word_freq)

    def score_final_output(seq):
        return len([1 for c,c2 in zip(seq, text) if c == c2])/float(len(text))
    print "Before spell check", score_final_output(seq)
    print "After spell check", score_final_output(spell_checked_seq)

'''
The following HMM code doesn't work as well as baum-welch and is slower.
This might be because of how we initialize the spaces though.


n_letters = len(valid_letters)
letters_typed = ['o', 'n' ,' ']  + [valid_letters[int(n_letters * random.random())] for i in range(n_clusters-3)]
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
'''
