import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import random
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from python_speech_features.features import mfcc
from scipy import signal

numclusters = 48

def load_data(wav_file, text_file):
    rate, data = scipy.io.wavfile.read(wav_file)
    #data_end = len(data) - 50000
    #data = np.abs(data[:data_end] + 0.01)

    data = data.astype(float)

    # Convert stereo to mono
    if len(data.shape) == 2 and data.shape[1] == 2:
        data = (data[:, 0] + data[:, 1])/2

    text = ''
    with open(text_file, 'r') as f:
        text = f.read()
    if text[-1] == '\n': text = text[:-1]
    return rate, data, text

def longest_common_subseq(list1, list2):
    # returns indices of each list which correspond to the LCS
    m,n = len(list1), len(list2)
    lcs_len = [[None for _ in range(n+1)] for _ in range(m+1)]
    lcs_action = [[None for _ in range(n+1)] for _ in range(m+1)] # used to reconstruct lcs
    for i in range(n+1):
        lcs_len[0][i] = 0
    for i in range(m+1):
        lcs_len[i][0] = 0

    for i in range(1,m+1):
        for j in range(1, n+1):
            options = [(lcs_len[i-1][j], "discard1"), (lcs_len[i][j-1], "discard2")]
            if list1[i-1] == list2[j-1]:
                options.append((lcs_len[i-1][j-1]+1, "match"))
            res = max(options)
            lcs_len[i][j] = res[0]
            lcs_action[i][j] = res[1]

    ci, cj = m, n
    list1_inds = []
    list2_inds = []
    while ci > 0 and cj > 0:
        ac = lcs_action[ci][cj]
        assert ac in ["discard1", "discard2", "match"]
        if ac == "discard1":
            ci -= 1
        elif ac == "discard2":
            cj -= 1
        else:
            assert list1[ci-1] == list2[cj-1]
            list1_inds.append(ci-1)
            list2_inds.append(cj-1)
            ci -= 1
            cj -= 1

    list1_inds.reverse()
    list2_inds.reverse()
    return list1_inds, list2_inds, np.array(list1)[list1_inds]

def plot_group(plot_set, title, *args):
    if title not in plot_set:
        return
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    plt.clf()
    for a, c in zip(args, colors):
        if isinstance(a, tuple):
            plt.plot(a[0], a[1], c)
        else:
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
    max_size = 62
    before_peak = 6
    win = np.ones(min_size)/float(min_size)
    p_sort = sorted(power)
    thresh1 = p_sort[int(0.95 * len(power))]
    thresh2 = p_sort[int(0.55* len(power))]
    thresh3 = p_sort[int(0.5 * len(power))]

    peaks = np.argwhere(power > thresh1)

    starts = []
    ends = []
    data_chunks=[]
    #plt.plot(np.log(power))
    #plt.plot(np.log(np.ones(len(power))*thresh1))
    #plt.plot(np.log(np.ones(len(power))*thresh2))
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

def clusterize(ls, spaces, soft_cluster, num_iterations=30, num_clusters=numclusters):
    print "using soft assignments:", soft_cluster
    bestclusters = 0
    bestmeans = 0
    bestscore = 1e1000000
    for i in range(num_iterations):
        print 'Try', i
        if soft_cluster:
            clusters, means, score = clusterize_inner_soft(ls, spaces, num_clusters)
        else:
            clusters, means, score = clusterize_inner(ls, spaces, num_clusters)
        if score < bestscore:
            bestmeans = means
            bestclusters = clusters
            bestscore = score
        print "Try", i, "score", score
    print "Best score:", bestscore
    return bestclusters, bestmeans

def clusterize_inner_soft(X, spaces, num_clusters=numclusters):

    # EM
    covar_type = "tied" #['spherical', 'diag', 'tied', 'full']
    classifier = GMM(n_components=num_clusters, covariance_type=covar_type,
            init_params='wc', n_iter=20)
    not_spaces = [i for i in range(len(X)) if i not in spaces]
    means = random.sample(X[spaces], 3) + random.sample(X[not_spaces], num_clusters-3)
    classifier.means_ = np.array(means)
    classifier.fit(X)

    #print "predict", classifier.predict(X)
    #print "means", classifier.means_
    #print "score", classifier.score(X)
    return classifier.predict_proba(X), classifier.means_, -np.sum(classifier.score(X))

    """
    This is soft k means, which doesn't work well

    def standardized(X):
        X_mean = np.mean(X,0)
        X_sd = np.std(X,0)
        return (X - X_mean) / X_sd

    m = len(X)
    X = standardized(X)

    feature_len = len(X[0])
    not_spaces = [i for i in range(len(X)) if i not in spaces]
    means = random.sample(X[spaces], 3) + random.sample(X[not_spaces], num_clusters-3)
    means = np.array(means)
    variances = np.ones((num_clusters, feature_len))

    Z = np.zeros((m, num_clusters))

    def compute_variances():
        variances = np.zeros((num_clusters, feature_len))
        for i in range(num_clusters):
            variances[i] = Z.T[i].dot((X - means[i])**2) / np.sum(Z.T[i])
        print "V:", variances
        return variances

    for j in range(20):
        # E step
        for i in range(m):
            dists = np.sum((means - X[i])**2 / variances, 1)
            #dists = dists - dists[0]
            #Z[i] = np.exp(-dists)
            Z[i] = 1/(dists + 0.1)
            Z[i] = Z[i] / np.sum(Z[i])
            #print "Z"
            #print Z[i]
            #print "dist"
            #print dists
            #print "sum"
            #print np.sum(Z[i])


        # M step
        means = Z.T.dot(X) / np.sum(Z,0)[:,np.newaxis]
        print "M", means
        variances = compute_variances()

    score = np.linalg.norm(X - np.dot(Z, means))
    return Z, None, score
    """

def clusterize_inner(ls, spaces, num_clusters=numclusters):
    '''Clusters the objects (np arrays) in ls into clusters
    The current algorithm is k-means
    '''
    def distance(a, b):
        return np.linalg.norm(a-b)

    n = len(ls[0])
    m = len(ls)
    notspaces = [i for i in range(len(ls)) if i not in spaces]
    means = random.sample(ls[spaces], 3) + random.sample(ls[notspaces], num_clusters-3)
    #variances = [np.identity(n) for _ in range(m)]
    assignments = [0 for i in range(len(ls))]
    dead_cluster = {}

    def var_sample(vect):
        v = vect.reshape((len(vect),1))
        return v.dot(v.T)

    def scaled_norm(vect, variance):
        v = vect.reshape((len(vect),1))
        return v.T.dot(np.linalg.inv(variance)).dot(v)

    def get_closest_cluster(means, l):
        smallest_i = -1
        smallest_e = 0
        for i, m in enumerate(means):
            if i in dead_cluster:
                continue
            #e = np.linalg.norm(l-m)
            e = distance(l, m)
            if e < smallest_e or smallest_i == -1:
                smallest_i = i
                smallest_e = e
        return smallest_i

    for j in range(40):

        changes = 0
        # E-step
        z = [[] for _ in range(num_clusters)]
        for (ind, l) in enumerate(ls):
            new_cluster = get_closest_cluster(means, l)
            z[new_cluster].append(l)
            if new_cluster != assignments[ind]:
                changes += 1
            assignments[ind] = new_cluster

        # M-step
        score = 0
        for i in range(num_clusters):
            if len(z[i]) == 0:
                dead_cluster[i] = True

            if i in dead_cluster:
                continue
            means[i] = sum(item for item in z[i])/len(z[i])
            for item in z[i]:
                score += np.linalg.norm(item - means[i])
            #variances[i] = sum(var_sample(item - means[i]) for item in z[i])/len(z[i])

        #print variances
        print "iteration:", j, "num_changes:", changes
        if changes < 5:
            break

    return np.array([get_closest_cluster(means, l) for l in ls]), means, score

def print_dict_sorted(d):
    l = []
    for v, k in sorted((v, k) for k, v in d.items()):
        l.append( '[%s]: %s' % (k, v))
    for i in range(0, len(l), 5):
        print '         '.join(l[i: i+5])

def baum_welch_inner(pi, theta, observations, spaces, text, soft_cluster, numclusters=numclusters):
    # Theta is transition probability: i-jth entry is transition from i to j
    K = len(pi) # Number of hidden states
    T = len(observations)
    phi = np.random.rand(len(pi), numclusters)
    phi[26, :] = 0
    for i in spaces:
        if soft_cluster:
            for clust_ind, prob in enumerate(i):
                phi[26, clust_ind] += prob / len(spaces)
        else:
            phi[26, i] += 1.0 / len(spaces)

    if soft_cluster:
        characteristic = observations
    else:
        characteristic = np.zeros((T, numclusters))
        for t, o in enumerate(observations):
            characteristic[t, o] = 1

    for iteration in range(150):
        # E-step
        # First, the forward-backward algorithm
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        #alpha[0, :] = pi * phi[:, observations[0]]
        alpha[0, :] = pi * phi.dot(observations[0])
        scale[0] = np.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :] / scale[0]
        for t in range(1, T):
            alpha[t, :] = np.dot(alpha[t-1, :], theta) * phi.dot(observations[t])
            scale[t] = np.sum(alpha[t, :])
            alpha[t, :] = alpha[t, :]/scale[t]

        beta = np.zeros((T, K))
        beta[T-1, :] = 1
        for t in range(T-2, -1, -1):
            beta[t, :] = np.dot(theta, phi.dot(observations[t+1]) * beta[t+1, :]) / scale[t+1]

        gamma = alpha * beta

        # M-step
        phi = (np.dot(gamma.transpose(), characteristic).transpose() / np.sum(gamma, axis=0)).transpose()
        #phi = (phi.transpose()/np.sum(phi, axis=1)).transpose()
        valid_letters = map(chr, range(97,123)) + [' ']

    seq = []

    for t in range(len(observations)):
        seq.append(valid_letters[np.argmax(gamma[t, :])])
    print ''.join(seq)
    print len([1 for s, c in zip(seq, text) if s == c])/float(len(seq))
    return phi, np.sum(np.log(scale))


def baum_welch(pi, theta, observations, spaces, text, soft_cluster):
    best_score = -1e1000000
    besti = 0
    bestphi = 0
    for i in range(15):
        print 'Tries', i
        phi, score = baum_welch_inner(pi, theta, observations, spaces, text, soft_cluster)
        if score > best_score:
            besti = i
            best_score = score
            bestphi = phi
    print best_score, besti
    return bestphi
