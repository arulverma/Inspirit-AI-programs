# This was my first experience programming a linear regression at the beginning of the AI course. 
# This code builds a linear regression model using Facebook data and estimates the amount of likes each type of post would get depending on various factors and info about the post itself
# This code is obviously not robust but it served as a healthy introduction into AI

import pandas as pd 
import os
import gdown

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split


gdown.download('https://drive.google.com/uc?id=1CL2so90NudcPDEiKaLAg02qQEGOJyI6o', 'facebook_data.csv', True)
data_path = 'facebook_data.csv'
fb_data = pd.read_csv(data_path,delimiter=";")
fb_data = fb_data.dropna()

sns.catplot(x = 'Type', y = 'like', data = fb_data, kind = 'swarm')
plt.show()
sns.scatterplot(x = 'like', y = 'share', data = fb_data)
plt.show()

fb_data['Post Category'] = fb_data['Type'].replace({'Photo':0 , 'Link':1, 'Status':2, 'Video':3})
x = fb_data[['comment','share','Paid','Post Category']].values
y = fb_data[['like']].values

m = linear_model.LinearRegression(fit_intercept=True , normalize= True)
m.fit(x,y)

x_test = [[2,1,4,5],[70,3,89,62]]
y_pred = m.predict(x_test)
print(y_pred.astype(int))
#print(metrics.accuracy_score(y_test,y_pred))
