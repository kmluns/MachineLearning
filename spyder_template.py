# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: kmluns
"""

import pandas as pd
import numpy as np


# %% all import
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# %% functions
def loadDataset():
    
    return load_boston()



# %% load dataset
data = loadDataset()
X = data['data']
y = data['target']
feature_names = data['feature_names']

#create the dataframe
boston_df = pd.DataFrame(X)
boston_df.columns = feature_names
boston_df.head()

# %% normalizing
X = (X - np.min(X)) / ( np.max(X) - np.min(X))

# %%
sns.boxplot(x=boston_df['DIS'])

# %% 
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['RAD'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()

# %%
plt.matshow(boston_df.corr())

# %% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)

pca.fit(X)
x_pca = pca.transform(X)


df_labelled = boston_df

df_labelled['class'] = y
df_labelled['p1'] = x_pca[:,0]
df_labelled['p2'] = x_pca[:,1]

plt.scatter(df_labelled.p1, df_labelled.p2, color='black' )
# =============================================================================
# plt.scatter(x_pca[:,0], x_pca[:,1], color='black' )
# =============================================================================
plt.legend()
plt.show()


# %% Z Score
z = np.abs(stats.zscore(X))
print(z)

threshold = 1.6
z_Threshold = np.where(z > 3)
print(z_Threshold)


# %% remove outlier Z Score
X_noOutlier = X[(z < threshold).all(axis=1)]


# %% IQR
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

IQR_result = ( X < (Q1 - 1.5 * IQR) ) | (X > (Q3 + 1.5 * IQR) )

# %% remove outlier IQR
X_notOutliers = X[~IQR_result.any(axis=1)]




# %%
y = data['target'][X_noOutlier.index]

# %% split test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_noOutlier, y, test_size=0.33, random_state=1)


# %% Random Forest Regressior
score_list = []
best_k = 3

for k in range(3,101):
    rf = RandomForestRegressor(k,random_state=1)
    rf.fit(X_train,y_train)
    score = rf.score(X_test,y_test)
    if len(score_list) > 0 and max(score_list) < score:
        best_k = k
    score_list.append(score)
    
print('score : {}'.format(score_list[best_k-3]))

# %% KNN 
from sklearn.neighbors import KNeighborsRegressor

score_list = []
best_k = 3

for k in range(3,101):
    kn = KNeighborsRegressor(k)
    kn.fit(X_train,y_train)
    score = kn.score(X_test,y_test)
    if len(score_list) > 0 and max(score_list) < score:
        best_k = k
    score_list.append(score)
    
print('score : {}'.format(score_list[best_k-3]))

