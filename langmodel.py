import re
import numpy as np
from mlalgs import print_dict_sorted

valid_letters = map(chr, range(97,123)) + [' ']
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

    counts = {c: counts[c]/float(sum(counts.values())) for c in counts}

    return counts, freq

def compute_bigram(fname = 'full_text.txt'):
    try:
        return np.loads('cache/next_letter_probs.npy')
    except Exception as e:
        next_letter_prob = {}

        with open("full_text.txt", "r") as f:
            s = f.read().lower()
            s = re.sub('[^a-z ]', ' ', s)
            s = re.sub(' +', ' ', s)
            letter_pairs = np.array(zip(s, s[1:]))
            pi, A = compute_freq_dist(letter_pairs)

            pi_v = np.zeros(len(valid_letters))
            theta_v = np.zeros((len(valid_letters), len(valid_letters)))

            for i, c in enumerate(valid_letters):
                pi_v[i] = pi[c]
                for j, c2 in enumerate(valid_letters):
                    theta_v[i, j] = A[c][c2]
            np.save('cache/next_letter_prob.npy', (pi, A, pi_v, theta_v))
            return pi, A, pi_v, theta_v

if __name__ == '__main__':
    pi, A, pi_v, theta_v = compute_bigram()
    print_dict_sorted(pi)
    print_dict_sorted(A)
