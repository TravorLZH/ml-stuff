#!/usr/bin/env python3
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

N=1e5

fs=10e3
noise_power=0.01*fs/2

time=np.arange(N)/float(fs)
noise=np.random.normal(scale=np.sqrt(noise_power),size=time.shape)
noise*=np.exp(-time/5)
series=np.sin(2*np.pi*3e3*time+500*np.cos(np.pi*time))+noise

f,t,Sxx=signal.spectrogram(series)

fig,(ax,ax2)=plt.subplots(2,1)
fig.set_tight_layout(True)
ax.plot(time,series)
ax2.pcolormesh(t,f,Sxx)
ax2.set_xlabel("Date")
ax2.set_ylabel("Frequency")
plt.show()
