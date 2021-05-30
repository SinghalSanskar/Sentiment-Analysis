# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/gdrive')

df=pd.read_csv('gdrive/My Drive/Dataset.csv')
df

train_dataset=df[:40000]
train_dataset

test_dataset=df[40000:]
test_dataset





import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw=set(stopwords.words('english'))

sw

import re
from nltk.stem import PorterStemmer
ps=PorterStemmer()

def clean(sample):
  sample=sample.lower()
  sample.replace("<br /><br />", "")
  sample=re.sub("[^a-zA-Z]+", " ",sample)
  sample=sample.split()

  sample=[ps.stem(s) for s in sample if s not in sw]
  sample=' '.join(sample)
  return sample

train_dataset['cleaned_review']=train_dataset['review'].apply(clean)

train_dataset

corpus=train_dataset['cleaned_review'].values

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
cv=CountVectorizer(max_df=0.5,max_features=50000)
x=cv.fit_transform(corpus)

tfidf=TfidfTransformer()
x=tfidf.fit_transform(x)

x.shape

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
y=train_dataset['sentiment'].values
y=label_encoder.fit_transform(y)
y[0:50]

from keras import models
from keras.layers import Dense

model=models.Sequential()
model.add(Dense(16, activation="relu",input_shape=(x.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy')

x_val=x[:5000]
x_train=x[5000:]
y_val=y[:5000]
y_train=y[5000:]

x_train.shape,y_train.shape

hist=model.fit(x_train,y_train,batch_size=128,epochs=2,validation_data=(x_val,y_val))

test_dataset['cleaned_review']=test_dataset['review'].apply(clean)

x_test=test_dataset['cleaned_review']

x_test=cv.transform(x_test)

x_test=tfidf.transform(x_test)

y_test=test_dataset['sentiment'].values
y_test=label_encoder.transform(y_test)

y_pred=model.predict(x_test)

y_pred

y_pred[y_pred>=0.5]=1
y_pred=y_pred.astype('int')

y_pred[0:10]

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))





















