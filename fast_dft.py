#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# This program is an implementation of Fast Fourier Transform
# Created by Travor Liu <travor_lzh@outlook.com> on 2020/4/22

# The original O(n^2) version of DFT
# a[m]: Time domain
# A[k]: Frequency domain
# N: Sample size
def dft(a):
    N=len(a)
    A=np.zeros(N,dtype=complex)
    for k in range(N):
        A[k]=a[0]
        for m in range(1,N):
            A[k]+=a[m]*np.exp(-2j*np.pi*k*m/N)
    return A

# This only works for samples with size 2^r
def fast_dft(a):
    N=len(a)
    if N==1:
        return a
    A=np.zeros(N,dtype=complex)
    B=fast_dft(a[::2])
    C=fast_dft(a[1::2])
    for k in range(0,N//2):
        tmp=np.exp(-2j*np.pi*k/N)
        A[k]=B[k]+tmp*C[k]
        A[k+N//2]=B[k]-tmp*C[k]
    return A

N=128
ts=np.linspace(0,5,N)
ys=np.sin(2*np.pi*ts)+2*np.sin(4*np.pi*ts)
freq=np.append(np.arange(0,N//2),(np.arange(0,N//2)-N//2))/N
amp=fast_dft(ys)

fig=plt.figure("FFT from scratch")
ax,ax2=fig.subplots(2,1)
fig.set_tight_layout(True)
ax.set_title("Time domain")
ax.plot(ts,ys,".-",label=r"$f(t)=\sin(2\pi t)+2\sin(4\pi t)$")
ax.set_xlabel("Time ($t$)")
ax.legend()

ax2.set_title("Frequency domain")
ax2.plot(freq,amp.real,label=r"$\Re[\hat{f}(\xi)]$")
ax2.plot(freq,amp.imag,label=r"$\Im[\hat{f}(\xi)]$")
ax2.set_xlabel(r"Frequency ($\xi$)")
ax2.legend()

plt.show()
