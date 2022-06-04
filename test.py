# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, style
style.use('seaborn-darkgrid')
import seaborn as sns
sns.set_style('darkgrid')
from plotly import express as px, graph_objects as go

from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor

import gc
gc.enable()
from warnings import filterwarnings, simplefilter
filterwarnings('ignore')
simplefilter('ignore')
rcParams['figure.figsize'] = (12, 9)

# %% [markdown]
# # 讀取資料

# %%
train = pd.read_csv('data/train.csv',parse_dates = ['date'], index_col = 'id')
event = pd.read_csv('data/holidays_events.csv',parse_dates = ['date'])
stores=pd.read_csv('data/stores.csv')
weekday=pd.read_csv('data/weekday.csv')
test=pd.read_csv('data/test.csv',parse_dates = ['date'], index_col = 'id')
submission=pd.read_csv('data/sample_submission.csv')

# %% [markdown]
# # 資料前處理

# %% [markdown]
# * 將train及test一起處理

# %%
train=pd.concat([train,test],axis=0)    
train
data=train.merge(event,how='left',on='date')  

# %% [markdown]
# * 增加星期幾的feature

# %%

#data["weeday"] = data["date"].dt.weekday
data.insert(2, 'weekday', data["date"].dt.weekday)
data.insert(2, "year", data["date"].dt.year)
data.insert(3, "date(mmdd)", data["date"].dt.month*100+data["date"].dt.day)

# %%
data

# %% [markdown]
# * 將日期是否為各類events做one hot encode

# %%
  
Holiday = pd.get_dummies(data.type, prefix='type')
transferred= pd.get_dummies(data.transferred, prefix='transferred')
data=data.drop(columns=['type','description','transferred','locale','locale_name'])
data=pd.concat([data,Holiday,transferred],axis=1)
#data["date"].dt.weekday
data

# %%
family=pd.get_dummies(data.family, prefix='family')
data=data.drop(columns=['family'])
data=pd.concat([data,family],axis=1)
data

# %%
data.describe()

# %%
data.drop(data[(data['type_Event'] ==1) ].index, inplace=True)
data.drop(["date"],axis='columns',inplace=True)

# %%
test=data.drop(data[(data.index <3000888) ].index)

data=data.drop(data[(data.index >=3000888) ].index)

# %%
data

# %%
rcParams['figure.figsize'] = (120,90)
sns.heatmap(data.corr().abs(), vmin=0, annot = True, fmt='.2f', vmax=1, linewidths=.3)

# %%


X = data.drop(['sales'],axis = 1)
y = data['sales']
X

# %%
y

# %%

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error as rmsle
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
DTR = DecisionTreeRegressor(random_state = 42)
DTR.fit(X_train,y_train)
predict_RFR = DTR.predict(X_test)

# %%
print(f"RMSLE: {(rmsle(y_test, predict_RFR))*100:0.2f}%")

# %%
test=test.drop(['sales','id'],axis = 1)


# %%
predict_RFR_sub = DTR.predict(test)

# %%
submission['sales'] = predict_RFR_sub
submission.to_csv('./submission.csv', index = False)
submission.head(10)


