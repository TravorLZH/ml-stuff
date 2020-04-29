#!/usr/bin/env python3
from chart_studio import plotly
import cufflinks as cf
import pandas as pd
import matplotlib.pyplot as plt

cf.go_offline()
cf.set_config_file(offline=False,world_readable=True)

data=pd.read_csv("baked_data.csv")
data.index=pd.to_datetime(data["DATE"])
data.drop("DATE",axis=1,inplace=True)
data.columns=["Energy Production"]

data.iplot()
#fig=data.plot()
#plt.show()
