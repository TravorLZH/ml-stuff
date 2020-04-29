#!/usr/bin/env python3
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

data=pd.read_csv("baked_data.csv")
data.index=pd.to_datetime(data["DATE"])
data.drop("DATE",axis=1,inplace=True)
data.columns=["Energy Production"]

try:
    forecast_df=pd.read_csv("forecast.csv")
    forecast_df.index=pd.to_datetime(forecast_df["DATE"])
    forecast_df.drop("DATE",axis=1,inplace=True)
except FileNotFoundError:
    # p,d,q are hyperparameters
    model=auto_arima(data,start_p=1,start_q=1,max_p=3,max_q=3,m=12,
            start_P=0,seasonal=True,d=1,D=1,trace=True,error_function="ignore",
            suppress_warnings=True,stepwise=True)
    # Only the data prior to 2010 are visible to the model
    train=data.loc["2000-01-01":"2010-12-01"]
    test=data.loc["2011-01-01":]

    model.fit(train)
    forecast=model.predict(n_periods=test.shape[0])

    forecast_df=pd.DataFrame(forecast,index=test.index,
            columns=["ARIMA Prediction"])
    forecast_df.to_csv("forecast.csv")

fig=pd.concat([data["2000-01-01":],forecast_df],axis=1).plot()
plt.legend()
plt.show()
