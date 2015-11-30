import sys
import math
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
import re
from supervised import logistic_test as logistic

from mlalgs import (
    load_data,
    get_chunk_starts,
    get_chunk_starts_simpler,
    clusterize,
    clusterize_inner,
    print_dict_sorted,
    plot_group,
    baum_welch,
    baum_welch_inner,
    longest_common_subseq,
    spell_check,
    hmm_predict,
    numclusters,
    generate_k_dist_changes
)

from features import (
    get_features, get_features_fft, get_features_cepstrum, get_features_energy
)

from langmodel import (
    valid_letters,
    compute_freq_dist,
    compute_bigram
)

def find_spaces(data, starts, ends):
    f = get_features_energy(data, starts, ends)
    clusts, means, _ = clusterize_inner(f, [np.argmax(f)], num_clusters=2)
    space_clust = 0
    if means[0] < means[1]:
        space_clust = 1
    spaces = [i for i in range(len(starts)) if clusts[i] == space_clust]
    return spaces

###################################################################
# Constants and debugging control
###################################################################

DATA_FILES = {
    'sound': 'data/sound7.wav', # sound1.wav
    'text': 'data/text7.txt', # text1.txt
}

rate, data, text = load_data(DATA_FILES['sound'], DATA_FILES['text'])

# the fraction of the file we use
file_range = (0, 1.0)

SOFT_CLUSTER = True
USE_PCA = True
DEBUG = False
# Which plots to actually plot.
PLOT_SET = set([
    #'Segmentation Plot',
    #'Features for Space',
    #'Features for %s' % text[0],
    #'Feature vector principal components',
])
PRINT_SET = set([
    'Segmentation',
    'Features',
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


pi, A, pi_v, theta_v, word_freq = cache_or_compute("cache/bigram.npy", compute_bigram,
        debug=False)

"""
spaces = []
for (i,c) in enumerate(text):
    if c == ' ':
        spaces.append(i)
        """

###################################################################
# Segmentation
###################################################################

starts, ends, chunks = cache_or_compute(
    'cache/chunks.npy', lambda arg: get_chunk_starts(arg),
    data[int(file_range[0] * len(data)): int(file_range[1] * len(data))], debug=False)

if not isinstance(starts, list):
    starts = starts.astype(int)
    ends = ends.astype(int)

if 'Segmentation' in PRINT_SET:
    print "num_chunks", len(chunks)

spaces = cache_or_compute("cache/spaces.npy", find_spaces, data, starts, ends)
'''
Cheating:
spaces = []
for (i,c) in enumerate(text):
    if c == ' ':
        spaces.append(i)
'''


'''
Spaciness code moved to spaces_code.py, since it is currently unused.
'''

N, M = 0, 1000000
ii1 = np.zeros(len(data))
ii1[starts] = 1
ii2 = np.zeros(len(data))
ii2[ends] = 1
plot_group(PLOT_SET, 'Segmentation Plot',
           np.log(np.abs(data[N:M]+0.01)),
           np.log(max(data[N:M]) * ii1[N:M]+0.01),
           np.log(max(data[N:M]) * ii2[N:M]+0.01),
           )

if CURRENT_STAGE == 'Segmentation':
    sys.exit()

###################################################################
# Feature Extraction
###################################################################

features = cache_or_compute('cache/features.npy', get_features, data, starts, ends, debug=False)

plot_group(PLOT_SET, 'Features for Space',
           features[spaces[0]])

plot_group(PLOT_SET, 'Features for %s' % text[0],
           features[0])
features = np.array(features)

if USE_PCA:
    pca = PCA(n_components=80)
    pca.fit(features)
    if 'Feature' in PRINT_SET:
        print "explained variances", pca.explained_variance_ratio_
        print "total explained var", sum(pca.explained_variance_ratio_)

    features = pca.transform(features)
    plot_group(PLOT_SET, 'Feature vector principal components',
               features[spaces[0]],
               features[0])

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
print "beginning clustering"
clusters, means = cache_or_compute('cache/clusters.npy', clusterize, features, spaces,
        SOFT_CLUSTER, pi_v, theta_v, text, debug=False)
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

spaces_bw = [clusters[i] for i in spaces]
phi, score, seq, gamma = cache_or_compute("hmm.npy", baum_welch, pi_v,
        theta_v, clusters, spaces_bw, text, SOFT_CLUSTER, 5, debug=False)
spell_checked_seq = spell_check(gamma, seq, word_freq)
print ''.join(spell_checked_seq)

def score_final_output(seq):
    return len([1 for c,c2 in zip(seq, text) if c == c2])/float(len(text))

print "Before spell check", score_final_output(seq)
print "After spell check", score_final_output(spell_checked_seq)

# Now, take output of spell checking, and use it to retrain hmm
counts = {}
for character, cluster in zip(spell_checked_seq, clusters):
    if character not in counts:
        counts[character] = np.ones(numclusters) * 0.5
    counts[character] += cluster

phi = np.random.rand(len(pi), numclusters)
valid_letters = map(chr, range(97,123)) + [' ']
for character in counts:
    phi[valid_letters.index(character), :] = counts[character]/np.sum(counts[character])

phi = (phi.transpose()/np.sum(phi, axis=1)).transpose()
phi, score, seq, gamma = baum_welch_inner(pi_v, theta_v, clusters, spaces_bw, text, SOFT_CLUSTER, phi=phi)

print '======================='
print 'After feedback'
spell_checked_seq = spell_check(gamma, seq, word_freq)
print ''.join(seq)
print score_final_output(seq)
print ''.join(spell_checked_seq)
print score_final_output(spell_checked_seq)

####################################################################
# Supervised on top of unsupervised
####################################################################

final_text = "".join(spell_checked_seq)
score, logreg = logistic(final_text, features)
ml_seq = logreg.predict(password_features)
letter_lls = logreg.predict_log_proba(password_features)

passwords = []
for num_changes in range(1,3):
    guesses = generate_k_dist_changes(ml_seq, num_changes, range(26))
    for seq in guesses:
        ll = sum([letter_lls[i] for i in seq])
        password = "".join(valid_letters[seq])
        passwords.append((ll,password))

passwords.sort()
