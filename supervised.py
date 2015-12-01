from features import get_features
from mlalgs import load_data, get_chunk_starts
from langmodel import valid_letters
import numpy as np
import scipy
import random
import sys

from sklearn import linear_model, svm
from sklearn.externals import joblib

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
valid_letters = list(valid_letters)
prior_v = np.zeros(len(valid_letters))
for c in prior:
    prior_v[list(valid_letters).index(c)] = prior[c]

def get_score(model, test_features, test_text):
    test_preds = model.predict(test_features)
    score = 0
    for l1,l2 in zip(test_preds, test_text):
        if l1 == l2:
            score += 1
    return score/float(len(test_preds))

def logistic_test(text, feature_v):
    '''Logistic regression, SVM'''

    assert len(text) == len(feature_v)
    text = list(text)
    feature_v = np.array(feature_v)
    cutoff = int(len(text) * 0.85)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(feature_v[:cutoff], text[:cutoff])
    test_features = feature_v[cutoff:]
    test_text = text[cutoff:]
    return get_score(logreg, test_features, test_text), logreg

def svm_test(text, feature_v):
    assert len(text) == len(feature_v)
    text = list(text)
    feature_v = np.array(feature_v)
    cutoff = int(len(text) * 0.85)
    rbf_svm = svm.SVC(kernel='rbf', gamma=0.7/len(feature_v[0]), C=1)
    rbf_svm.fit(feature_v[:cutoff], text[:cutoff])
    test_features = feature_v[cutoff:]
    test_text = text[cutoff:]
    return get_score(rbf_svm, test_features, test_text), rbf_svm

def logistic(text, feature_v, letters=valid_letters):
    feature_v = np.array(feature_v)
    pairing = zip(text, feature_v)
    examples = [[] for _ in letters]

    cutoff = int(len(text) * 0.8)
    training = pairing[:cutoff]
    test = training

    weights = {c: 1.0 / len(feature_v[0]) * np.random.randn(feature_v.shape[1] + 1) for c in letters}

    def sigmoid(w, x):
        '''Numerically stable sigmoid function'''
        agg = np.dot(x, w)
        sig1 = 1.0/(1+np.exp(-agg))
        z = np.exp(agg)
        sig2 = z/(1+z)
        sig = sig1
        if len(agg.shape) == 0:
            if agg < 0:
                return sig2
            return sig1
        w = agg<0
        sig[w] = sig2[w]
        return sig


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
        random.shuffle(negatives)
        negatives = np.array(negatives)
        def err(weights):
            sig = sigmoid(weights, positives)
            diff = np.sum((1-sig)**2)
            sig = sigmoid(weights, negatives)
            diff += np.sum((0-sig)**2)
            return diff
        def derr(weights):
            sig = sigmoid(weights, positives)
            deriv = sig * (1-sig)
            diff = np.sum(2*(sig-1)*deriv* positives.transpose(), axis=1)
            sig = sigmoid(weights, negatives)
            deriv = sig * (1-sig)
            diff += np.sum(2*(sig)*deriv* negatives.transpose(), axis=1)
            return diff
        res=scipy.optimize.minimize(err, weights[c], jac=derr, method='BFGS', options={'maxiter': 400})
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

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: %s training|test soundf textf' % sys.argv[0]

    soundf = sys.argv[2]
    textf = sys.argv[3]

    rate, data, text = load_data(soundf, textf)
    starts, ends, chunks = get_chunk_starts(data)
    f = get_features(data, starts, ends, include_fft=True, include_cepstrum=True)

    if sys.argv[1] == 'training':
        means, stds, score = naive_bayes(text, f)
        print 'Naive Bayes', score

        logreg_score, logreg = logistic_test(text, f)
        svm_score, svm = svm_test(text, f)
        joblib.dump(logreg, 'cache/logistic.pkl')
        print 'Logistic test', logreg_score
        print 'SVM test', svm_score
    else:
        try:
            logreg = joblib.load('cache/logistic.pkl')
        except:
            print 'Run `%s training 7` to train your model first' % sys.argv[0]
            sys.exit()
        print get_score(logreg, f, text)
        print ''.join(logreg.predict(f))
