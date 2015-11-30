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
    clusterize_inner,
    print_dict_sorted,
    plot_group,
    baum_welch,
    longest_common_subseq,
    hmm_predict
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

pi, A, pi_v, theta_v = compute_bigram()

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
        #np.save(fname, c)
        return c
    try:
        print "using cache at", fname, "for", fun
        return np.load(fname)
    except IOError:
        c = fun(*args)
        np.save(fname, c)
        return c

rate, data, text = load_data(DATA_FILES['sound'], DATA_FILES['text'])

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

spaces = find_spaces(data, starts, ends)
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
           -np.log(max(data[N:M]) * ii2[N:M]+0.01),
           )

if CURRENT_STAGE == 'Segmentation':
    sys.exit()

###################################################################
# Feature Extraction
###################################################################

features = cache_or_compute('cache/features.npy', get_features, data, starts, ends, debug=True)

plot_group(PLOT_SET, 'Cepstrum Plot',
           features[spaces[0]],
           features[spaces[1]],
           features[0], features[1])
features = np.array(features)

if USE_PCA:
    pca = PCA(n_components=30)
    pca.fit(features)
    if 'Feature' in PRINT_SET:
        print "explained variances", pca.explained_variance_ratio_
        print "total explained var", sum(pca.explained_variance_ratio_)

    features = pca.transform(features)
    plot_group(PLOT_SET, 'PCA Plot',
               features[spaces[0]],
               features[spaces[1]],
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

print "beginning clustering"
clusters, means = cache_or_compute('cache/clusters.npy', clusterize, features, spaces,
        SOFT_CLUSTER, pi_v, theta_v, text, debug=True)
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
phi, _ = baum_welch(pi_v, theta_v, clusters, spaces_bw, text, SOFT_CLUSTER, 15)
predicted = hmm_predict(pi_v, theta_v, phi, clusters)

print '==============================='
print '==============================='
print '==============================='
print '==============================='
print predicted
