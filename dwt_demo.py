#!/usr/bin/env python3
import pywt
import matplotlib.pyplot as plt
import numpy as np

ts=np.linspace(0,5,128)
samples=np.sin(2*np.pi*ts)+2*np.sin(4*np.pi*ts)

ca,cd=pywt.dwt(samples,"db1")
freq=range(len(ca))

fig=plt.figure("FFT from scratch")
ax,ax2=fig.subplots(2,1)
fig.set_tight_layout(True)
ax.set_title("Time domain")
ax.plot(ts,samples,".-",label=r"$f(t)=\sin(2\pi t)+2\sin(4\pi t)$")
ax.set_xlabel("Time ($t$)")
ax.legend()

ax2.set_title("Frequency domain")
ax2.plot(freq,ca,label=r"Approximation coefficient")
ax2.plot(freq,cd,label=r"Detail coefficient")
ax2.legend()

plt.show()
