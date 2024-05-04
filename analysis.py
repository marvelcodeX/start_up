import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import sklearn


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score, accuracy_score,classification_report, confusion_matrix, precision_score, recall_score

df = pd.read_csv(r"C:\Users\Jay\Downloads\startupdata2.csv")

#df = pd.read_csv("C:\Users\Jay\Downloads\startupdata2.csv")

clist = pd.Series(df.isnull().sum()[df.isnull().any()].sort_values(ascending = False))

clist = list(clist.index)
clist = clist[:-1]

cols_to_drop = ['id','Unnamed: 0','Unnamed: 6','closed_at','state_code.1','labels', 'zip_code', 'longitude','latitude', 'object_id']
df.drop(cols_to_drop,axis=1,inplace=True)

df['status'] = df['status'].replace({'acquired':1, 'closed':0})

df['founded_at'] = pd.to_datetime(df['founded_at'])
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'])
df['last_funding_at'] = pd.to_datetime(df['last_funding_at'])

df.drop_duplicates(inplace=True)

df_num=df.select_dtypes(include=np.number)

plt.figure(figsize = (25, 18))
sns.heatmap(df_num.corr(), annot = True, cmap = 'coolwarm', linewidth = 0.5, fmt = '.1f')

plt.title('Chart')
plt.show()