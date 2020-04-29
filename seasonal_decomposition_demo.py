#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#data=pd.read_csv("Electric_Production.csv")
#data=data["1980-01-01":]
#data.to_csv("baked_data.csv")

data=pd.read_csv("baked_data.csv")
data.index=pd.to_datetime(data["DATE"])
data.drop("DATE",axis=1,inplace=True)
data.columns=["Energy Production"]

result=seasonal_decompose(data,model="multiplicative")
fig=result.plot()
plt.show()
