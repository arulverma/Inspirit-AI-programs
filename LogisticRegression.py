import pandas as pd 
import os
import gdown

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split


gdown.download('https://drive.google.com/uc?id=12WcGMubUMHGuS5-wj3PadyQIsJiZZTcd', 'heart_data.csv', True)

data_path  = 'heart_data.csv'
heart_data = pd.read_csv(data_path)
heart_data = heart_data.dropna() # remove rows with missing values

sns.catplot(x = 'target' , y = 'oldpeak', data = heart_data)
plt.show()

sns.scatterplot(x = 'trestbps' , y = 'chol' , data = heart_data)
plt.show()

train_df, test_df = train_test_split(heart_data, test_size = 0.4, random_state = 1)
input_labels = ['age','sex','cp','trestbps','chol','fbs','exang','oldpeak','ca','thal']
output_labels = 'target'
x_train = train_df[input_labels]
y_train = train_df[output_labels]
x_test = test_df[input_labels]
y_test = test_df[output_labels]

m = linear_model.LogisticRegression(max_iter = 10000)
m.fit(x_train,y_train,sample_weight = None)

y_pred = m.predict(x_test)
print(y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
