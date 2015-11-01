import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)

rate, data = scipy.io.wavfile.read("long_test.wav")

def get_cluster_starts(data):
    kernel = np.ones(20)

plt.plot(data)
plt.show()
