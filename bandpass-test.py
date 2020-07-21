# Plot frequency response of the bandpass for QA

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Define the bandpass filter
nyquist = 44100 * 0.5
lowcut = 28.0/nyquist
highcut = 17500.0/nyquist 
sos = signal.butter(3, [lowcut, highcut], btype='bandpass', output='sos')

# Plot the frequency response
w, h = signal.sosfreqz(sos, worN=16384)
plt.plot((nyquist/np.pi) * w, abs(h))

# Zoom in on the highpass portion of the filter
plt.xlim(0,100)
plt.ylim(0,1.2)

plt.show()