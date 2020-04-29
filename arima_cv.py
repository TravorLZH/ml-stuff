#!/usr/bin/env python3
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("baked_data.csv")
data.index=pd.to_datetime(data["DATE"])
data.drop("DATE",axis=1,inplace=True)

train=data.loc["1970-01-01":"2000-12-01"]
validation=data.loc["2000-01-01":"2010-01-01"]
test=data.loc["2010-01-01":]

validation_y=np.array(validation).flatten()
nV=len(validation_y)
test_y=np.array(test).flatten()
nT=len(test_y)

# range of p, d, and q
P=np.arange(1,4)
D=np.arange(1,4)
Q=np.arange(1,4)
# range of seasonal parameters (p,d,q,s)
sP=np.arange(0,4)
sD=np.arange(0,4)
sQ=np.arange(0,4)
M=np.arange(2,12+1)

stochastic=True
ncandidates=20

mesh=[a.flatten() for a in np.meshgrid(P,D,Q,sP,sD,sQ,M)]
#pv=np.ones_like(mesh[0],dtype=np.int)*2
#dv=np.ones_like(mesh[0],dtype=np.int)
#qv=np.ones_like(mesh[0],dtype=np.int)
#mv=np.ones_like(mesh[0],dtype=np.int)*6
paramsV=np.array([*mesh]).T
if stochastic==True:
    np.random.shuffle(paramsV)
    paramsV=paramsV[:ncandidates]
print(paramsV.shape)

params_final=np.zeros_like(paramsV[0])
least_err=5000
test_err=0
yhat=[]

for params in paramsV:
    print("Training ARIMA(%d,%d,%d) seasonal=(%d,%d,%d,%d)" % tuple(params))
    model=pm.ARIMA(order=tuple(params[:3]),seasonal_order=tuple(params[3:]),
            suppress_warnings=True)
    try:
        model.fit(train)
    except Exception as ex:
        print("Error occurred: %s" % ex)
        continue
    y=model.predict(n_periods=nV+nT)
    err=np.square(y[:nV]-validation_y).mean()
    print("MSE=%.5f" % err)
    if least_err>err:
        least_err=err
        yhat=y
        params_final=params
        test_err=np.square(yhat[nV:]-test_y).mean()

print("Best hyperparameters: ARIMA(%d,%d,%d) seasonal=(%d,%d,%d,%d)" \
        " with validation MSE=%.5f and testing MSE=%.5f"
        % (*params_final,least_err,test_err))

forecast_df=pd.DataFrame(yhat[:nV],index=validation.index)
forecast_df.columns=["Model prediction on validation set"]
forecast2_df=pd.DataFrame(yhat[nV:],index=test.index)
forecast2_df.columns=["Model prediction on test set"]
fig=pd.concat([data,forecast_df,forecast2_df],axis=1).plot()
plt.title("Prediction of $ARIMA(%d,%d,%d)(%d,%d,%d)_{%d}$"
        % tuple(params_final))
plt.legend()
plt.show()
