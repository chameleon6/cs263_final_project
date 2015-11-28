from features import get_features
from mlalgs import load_data, get_chunk_starts
from langmodel import valid_letters
import numpy as np
import scipy

prior = {' ': 0.17754225405572727, 'a': 0.0661603239798244, 'b':
         0.012621361579079272, 'c': 0.025518818692412935, 'd':
         0.03262728212110697, 'e': 0.10277443948100279, 'f':
         0.01918733974586732, 'g': 0.016037285549336523, 'h':
         0.044719330604384044, 'i': 0.05992936945880769, 'j':
         0.0013241493293807592, 'k': 0.005405441005854335, 'l':
         0.03399944052957654, 'm': 0.02091376009533875, 'n':
         0.05836118842055676, 'o': 0.062448199590206976, 'p':
         0.016630033024006965, 'q': 0.0008844416266793212, 'r':
         0.050427384794675394, 's': 0.053851628027735035, 't':
         0.07608295136940266, 'u': 0.02232422177629433, 'v':
         0.008193087316555922, 'w': 0.01545147079366807, 'x':
         0.001635948366495417, 'y': 0.014166144690698474, 'z':
         0.0007827039753250665}

prior_v = np.zeros(len(valid_letters))
for c in prior:
    prior_v[valid_letters.index(c)] = prior[c]

def logistic(text, feature_v, letters=valid_letters):
    feature_v = np.array(feature_v)
    pairing = zip(text, feature_v)
    examples = [[] for _ in letters]

    cutoff = int(len(text) * 0.8)
    training = pairing[:cutoff]
    test = pairing[cutoff:]

    weights = {c: np.random.randn(feature_v.shape[1] + 1) for c in letters}

    def sigmoid(w, x):
        return 1.0/(1+np.exp(-np.dot(w, x)))

    examples = {c: [] for c in letters}
    for c, f in training:
        examples[c].append(np.concatenate([f, np.array([1])]))

    for c in examples:
        if len(examples[c]) == 0: continue
        print len(examples[c])
        positives = np.array(examples[c])
        negatives = []
        for d in examples:
            if c == d or len(examples[d]) == 0: continue
            negatives.extend(examples[d])
        negatives = np.array(negatives)
        def err(weights):
            agg = np.dot(positives, weights)
            sig = 1.0/(1+np.exp(-agg))
            diff = np.sum((1-sig)**2)
            agg = np.dot(negatives, weights)
            sig = 1.0/(1+np.exp(-agg))
            diff += np.sum((0-sig)**2)
            return diff
        def derr(weights):
            agg = np.dot(positives, weights)
            sig = 1.0/(1+np.exp(-agg))
            deriv = sig * (1-sig)
            diff = np.sum(2*(sig-1)*deriv* positives.transpose(), axis=1)
            agg = np.dot(negatives, weights)
            sig = 1.0/(1+np.exp(-agg))
            deriv = sig * (1-sig)
            diff += np.sum(2*(sig)*deriv* negatives.transpose(), axis=1)
            return diff
        res=scipy.optimize.minimize(err, weights[c], jac=derr, method='BFGS', options={'maxiter': 100})
        weights[c] = res.x

    score = 0
    pred = []
    real = []
    for c, f in test:
        f = np.concatenate([f, np.array([1])])
        maxc = ''
        maxv = 0
        for d in weights:
            m = sigmoid(weights[d], f)
            if m > maxv:
                maxv = m
                maxc = d
        pred.append(maxc)
        real.append(c)
        if c == maxc:
            score += 1
    print ''.join(pred)
    print ''.join(real)
    return weights, score/float(len(test))

def naive_bayes(text, feature_v, letters=valid_letters):
    '''Train a classifier to distinguish between ' ', 'e', and 't'
    and output accuracy of classifier'''
    feature_v = np.array(feature_v)
    pairing = zip(text, feature_v)

    means = np.zeros((len(letters), feature_v.shape[1]))
    stds = np.zeros((len(letters), feature_v.shape[1]))
    examples = [[] for _ in letters]

    cutoff = int(len(text) * 0.8)
    training = pairing[:cutoff]
    test = pairing[cutoff:]
    for c, f in training:
        if c not in letters:
            continue
        i = letters.index(c)
        examples[i].append(f)

    for i in range(len(letters)):
        means[i] = np.mean(np.array(examples[i]), axis=0)
        stds[i] = np.std(np.array(examples[i]), axis=0)

    score = 0
    pred = []
    real = []
    for c, f in test:
        if c not in letters:
            continue
        minj= 0
        minscore = -1e100000
        for j in range(len(letters)):
            s = -np.sum((f - means[j])**2/stds[j]**2) + np.log(prior[letters[j]])
            if s > minscore:
                minj = j
                minscore = s
        pred.append(letters[minj])
        real.append(c)
        if minj == letters.index(c):
            score += 1
    print ''.join(pred)
    print ''.join(real)
    return means, stds, score/float(len(test))

rate, data, text = load_data('data/sound7.wav', 'data/text7.txt')
starts, ends, chunks = get_chunk_starts(data)

f = get_features(data, starts, ends)
means, stds, score = naive_bayes(text, f)

print 'Naive Bayes', score

weights, score = logistic(text, f)
print 'Logistic', score