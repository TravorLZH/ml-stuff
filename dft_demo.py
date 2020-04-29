#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def signal(t):
    return np.sin(2*np.pi*t)+2*np.sin(4*np.pi*t)

ts=np.linspace(0,5,128)
ts2=np.linspace(-5,5,100)
samples=signal(ts)
samples_rect=np.where(abs(ts2)<=0.5,1,0)

amp=np.fft.fft(samples)
amp_rect=np.fft.fft(samples_rect)
freq=np.fft.fftfreq(len(ts))
freq2=np.fft.fftfreq(len(ts2))

fig=plt.figure("DFT & DWT Demo")
fig.set_tight_layout(True)

(td,td2),(fd,fd2)=fig.subplots(2,2)

td.set_title("Time domain $f(t)=\sin(2\pi t)+2\sin(4\pi t)$")
td.plot(ts,samples,label=r"$f(t)$")
td.legend()

fd.set_title(r"Frequency domain $F(\omega)=\mathcal{F}\{f(t)\}$")
fd.plot(freq,amp.real,label=r"$\Re[F(\omega)]$")
fd.plot(freq,amp.imag,label=r"$\Im[F(\omega)]$")
fd.legend()

td2.set_title(r"Time domain $\mathrm{rect}(t)$")
td2.plot(ts2,samples_rect,label=r"$\mathrm{rect}(t)=\frac{\sin(\pi t)}{\pi t}$")
td2.legend()

fd2.set_title(r"Frequency domain " \
        r"$F(\omega)=\mathcal{F}\{\mathrm{rect}(t)\}$")
fd2.plot(freq2,amp_rect.real,label=r"$\Re[F(\omega)]$")
fd2.plot(freq2,amp_rect.imag,label=r"$\Im[F(\omega)]$")
fd2.legend()

plt.show()
