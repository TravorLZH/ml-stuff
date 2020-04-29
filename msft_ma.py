#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import stats

msft=pd.read_csv("msft.csv")

xs=range(0,len(msft))
avg=np.mean(np.array([msft.Low,msft.High]),axis=0)

# Now find the line of best fit
#a,b,r,p,err=stats.linregress(xs,avg)
#ln=a*xs+b

# Find a polynomial fit
#coeffs=np.polyfit(xs,avg,3)
#Xs=np.array([np.power(xs,3),np.square(xs),xs,np.ones_like(xs)])
#ln=np.dot(coeffs,Xs)

# Find moving average
win_size=5
ma_range=range(0,len(xs)-win_size)
ma=[]
for i in ma_range:
    ma.append(np.mean(avg[i:i+win_size]))

noise=avg[0:len(xs)-win_size]-ma

def setup_plot(no,tit):
    p=plt.subplot(no)
    p.set_xlabel("Time (days)")
    p.set_ylabel("Stock Price ($)")
    p.set_title(tit)
    return p

fig=plt.figure(figsize=(12,8))
fig.set_tight_layout(True)
#plt.plot(xs,msft.Low,".-",label="Daily low")
#plt.plot(xs,msft.High,".-",label="Daily high")
ax=setup_plot(221,"MSFT stock price from %s to %s" %
        (msft.Date[0],msft.Date[xs[-1]]))
ax.plot(xs,avg,".-",label="Daily (average) stock price")
ax.plot(ma_range,ma,label="Moving averages with window size %d" % win_size)
ax.legend()
ax2=setup_plot(222,"Noise (Daily stock price minus moving averages)")
ax2.plot(ma_range,noise,"r.-")
# Now generate a set of Gaussian noise around the line
newnoise=np.random.normal(loc=np.mean(noise),scale=np.std(noise),
        size=len(ma))
ax3=setup_plot(223,"Generated Gaussian noise")
ax3.plot(ma_range,newnoise,"r.-")
ax4=setup_plot(224,"Moving averages + Generated Gaussian noise")
ax4.plot(xs,avg,".-",label="Original data")
ax4.plot(ma_range,ma+newnoise,".-",label="Predicted data")
ax4.legend()

#plt.show()
plt.savefig("MSFT_%d.png" % win_size)
