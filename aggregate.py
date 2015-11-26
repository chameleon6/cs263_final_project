'''
Usage: ./aggregate outname in_wavfiles
Example: ./aggregate 7 sound7.*.wav
will aggregate sound files
sound7.1.wav sound7.2.wav, etc
with corresponding text files
text7.1.txt text7.2.txt
The output aggregation will be guaranteed to have perfect segmentation
'''
import sys
from mlalgs import (load_data, get_chunk_starts)
import scipy.io.wavfile
import numpy as np

tot = []
tot_txt = []
tot_chars = 0
for f in sys.argv[2:]:
    text_file = 'text' + f[5:-3] + 'txt'
    rate, data, text = load_data(f, text_file)
    starts, _, _ = get_chunk_starts(data)
    if len(starts) != len(text):
        print '%s rejected: %d != %d' % (f, len(starts), len(text))
        continue
    tot_chars += len(text)
    tot.append(data)
    tot_txt.append(text)

print 'Created data file with %d characters' % tot_chars
scipy.io.wavfile.write('sound%s.wav' % sys.argv[1], rate, np.concatenate(tot))
with open('text%s.txt' % sys.argv[1], "w") as f:
    f.write(''.join(tot_txt))
