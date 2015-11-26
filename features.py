from python_speech_features.features import mfcc
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from mlalgs import (load_data, get_chunk_starts)

valid_letters = tuple(map(chr, range(97,123)) + [' '])

def binom_kernel(n):
    kernel = [1.0]
    for i in range(n):
        k1 = [0]
        k1.extend(kernel)
        k2 = kernel[:]
        k2.append(0)
        kernel = [(x+y)/2 for x,y in zip(k1, k2)]

    return np.array(kernel)

def normalize(f):
    return (f - np.mean(f, axis=0)) / np.std(f, axis=0)

def get_features_fft(data, starts, ends, kernel_len=100, lower=0, upper=3500):
    kernel_len = 100
    features = []
    kernel = binom_kernel(kernel_len)
    for i,e in zip(starts, ends):
        spec = np.absolute(np.fft.fft(data[i:e])[lower: upper])
        f = np.convolve(spec, kernel)
        features.append(f[::kernel_len])
    return normalize(np.log(np.abs(features)+0.001))

def get_features_cepstrum(data, starts, ends,
                         winlen=0.01, winstep=0.0025,
                         numcep=16, nfilt=32, appendEnergy=False, keylen=20):
    '''Grab the features from a single keystroke'''
    # return np.float64(np.log(np.absolute(np.float32(chunk))+0.01)[:4000:4])
    # The definition of cepstrum
    f = mfcc(data, samplerate=44100, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, appendEnergy=appendEnergy)
    ans = []
    print len(f)
    print len(data)/(44100*winstep)
    for s in starts:
        max_s = int(0.005/(winstep) * keylen)
        b = int(s / (44100*winstep))
        if b+max_s >= len(f): break
        tot = []
        for i in range(b, b+max_s):
            tot.append(f[i])
        ans.append(np.concatenate(tot))
    return ans
    return np.concatenate((f[0], f[1], f[2]))
    f=np.absolute(np.fft.fft(chunk, n=256))
    lf = np.log(f ** 2)
    c = (np.absolute(np.fft.ifft(lf))**2)
    return normalize(np.log(c[0:50]))

def get_features(data, starts, ends):
    f1 = get_features_fft(data, starts, ends)
    f2 = get_features_cepstrum(data, starts, ends)
    print "fft feature len:", len(f1[0])
    print "cepstrum feature len", len(f2[0])
    f = np.array([np.append(x, y) for x,y in zip(f1,f2)])
    return f

def score(text, feature_v, letters=valid_letters):
    '''Train a classifier to distinguish between ' ', 'e', and 't'
    and output accuracy of classifier'''
    feature_v = np.array(feature_v)
    pairing = zip(text, feature_v)[:int(0.2*len(text))]
    # Empirical letter frequencies
    prior = {' ': 0.17754225405572727, 'a': 0.08044221339122587, 'c': 0.033741774654502114, 'b': 0.01727111419151551, 'e': 0.14310853647395913, 'd': 0.05301948291262867, 'g': 0.027519744707964172, 'f': 0.03385692474818912, 'i': 0.10945383955813408, 'h': 0.0917128383239704, 'k': 0.012205140432923142, 'j': 0.003026786929778515, 'm': 0.04795054157202014, 'l': 0.08187921049634359, 'o': 0.16380304097600146, 'n': 0.18307015217871747, 'q': 0.0033960789146692287, 'p': 0.0640735988417217, 's': 0.22168847983899614, 'r': 0.2667209853900528, 'u': 0.1610266167506027, 't': 0.6541244823613469, 'w': 0.38408012315480666, 'v': 0.3306543219302994, 'y': 0.854134628541169, 'x': 0.6761849057341293, 'z': 0.9989215169463295}

    s_prior = sum(prior[c] for c in letters)
    for c in letters:
        prior[c] /= s_prior

    means = np.zeros((len(letters), feature_v.shape[1]))
    stds = np.zeros((len(letters), feature_v.shape[1]))
    examples = [[] for _ in letters]
    for c, f in pairing:
        if c not in letters:
            continue
        i = letters.index(c)
        examples[i].append(f)

    for i in range(len(letters)):
        means[i] = np.mean(np.array(examples[i]), axis=0)
        stds[i] = np.std(np.array(examples[i]), axis=0)

    score = 0
    total = 0
    for i in range(len(letters)):
        for f in examples[i]:
            total += 1
            minj= 0
            minscore = -1e100000
            # import pdb; pdb.set_trace()
            for j in range(len(letters)):
                s = np.sum(-(f - means[j])**2/stds[j]**2) - np.log(prior[letters[j]])
                if s > minscore:
                    minj = j
                    minscore = s
            if minj == i:
                score += 1
    return score/float(total)

def visualize(text, feature_v, ind1, ind2, letters=(' ', 'e', 't')):
    colormap = ['r', 'g', 'b', 'm', 'c', 'k']
    pairing = zip(text, feature_v)[:int(0.16*len(text))]
    xs = []
    ys = []
    colors = []
    for c, f in pairing:
        if c not in letters:
            continue
        i = letters.index(c)
        xs.append(f[ind1])
        ys.append(f[ind2])
        colors.append(colormap[i])
    plt.scatter(xs, ys, c=colors)
    plt.show()

def visualize_segmentation(data, starts, ends):
    starts_m = np.zeros(len(data))
    ends_m = np.zeros(len(data))
    starts_m[starts] = 1
    ends_m[ends] = 1

    plt.plot(np.log(np.abs(data)+0.01), 'r')
    plt.plot(np.log(max(data)) * starts_m, 'g')
    plt.plot(-np.log(max(data)) * ends_m, 'b')
    plt.show()

def gen_segmentation_report(data, starts, ends):
    pdf_pages = PdfPages('segments.pdf')
    starts_m = np.zeros(len(data))
    ends_m = np.zeros(len(data))
    starts_m[starts] = np.log(max(data))
    ends_m[ends] = -np.log(max(data))
    step = 44100*5
    counter = 0
    prev_counter = 0
    for i, t in enumerate(range(0, len(data), step)):
        if i % 3 == 0:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.subplot2grid((3, 1), (i % 3, 0))

        plt.plot(np.log(np.abs(data[t: t+step])+0.01), 'r')
        plt.plot(starts_m[t: t+step], 'g')
        plt.plot(ends_m[t: t+step], 'b')

        while counter < len(ends) and ends[counter] < min(t + step, len(data)):
            counter += 1
        counter -= 1
        plt.title('Chars %s to %s' % (prev_counter, counter))
        prev_counter = counter
        
        if (i+1) % 3 == 0:
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()


'''
rate, data, text = load_data('data/sound6.wav', 'data/text5.txt')
starts, ends, chunks = get_chunk_starts(data)
'''
