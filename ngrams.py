import numpy as np

next_letter_prob = {}
counts = {}
valid_letters = map(chr, range(97,123)) + [' ']
with open("full_text.txt", "r") as f:
    s = f.read()
    for c1, c2 in zip(s, s[1:]):
        if c1 not in valid_letters or c2 not in valid_letters:
            continue
        if c1 not in counts:
            counts[c1] = 0
            next_letter_prob[c1] = {}
        if c2 not in next_letter_prob[c1]:
            next_letter_prob[c1][c2] = 0
        next_letter_prob[c1][c2] += 1
        counts[c1] += 1

for c1 in next_letter_prob:
    for c2 in next_letter_prob:
        if c2 not in next_letter_prob[c1]:
            next_letter_prob[c1][c2] = 0

for c1 in next_letter_prob:
    for c2 in next_letter_prob[c1]:
        next_letter_prob[c1][c2] /= float(counts[c1])

np.save("cache/next_letter_prob.npy", next_letter_prob)
