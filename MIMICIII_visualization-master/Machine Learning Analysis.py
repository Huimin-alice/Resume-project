#!/usr/bin/env python
# coding: utf-8

# # Analysis of MIMIC

# ## Huimin Jiang (hj2532)

# ## Step 1 Data Cleaning
# 

# In[ ]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest,SelectPercentile,SelectFpr
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import f_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel


# In[ ]:


#read data from csv
df=pd.read_csv("COMB.csv")
df=df[df['age']<120]
df=df.drop(['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ADMITTIME','DEATHTIME','mortality_time','DOB','AGE'],axis=1)
df.shape
df.info()


#     By reading the description and info of dataset, we find every item's feature `url` and `description` is a unique value, they are hard to be classified into different classes. Similarly, the `image_url`,`vin`,`model`,`region` and `region_url` have too many unique categorical variables so we simply drop them. Also, we remove the `id` beacause it will leak target information. Besides, since it is difficult to explain the meaning of 0 price, we remove all the columns with price 0.
#     Also, we drop the outliners which is larger than 99% of the most data.

# In[ ]:


df.target=df['mortality_withinthirtydays']
df.data=df.drop(['mortality_withinthirtydays'],axis=1)

categoricals = df.data.select_dtypes(include='object')
numerics = df.data._get_numeric_data()
df.data.columns


# In[ ]:


sns.pairplot(numerics,height=5)


# ## Step 2 Baseline model

#     We use ridge as base model here.

# In[ ]:


X=df.data
y=df.target.values

categorical = X.dtypes==object

categoric_pre= make_pipeline(SimpleImputer(strategy='constant',fill_value='missing_value'),OneHotEncoder(handle_unknown='ignore'))
numeric_pre= make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

preprocess= make_column_transformer((categoric_pre,categorical),(numeric_pre,~categorical))
rg_pipe=make_pipeline(preprocess,Ridge())
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
rg_pipe.fit(X_train,y_train)
scores = cross_val_score(rg_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)


print("Baseline model train cross valid score:")
print(mean_score)
print("Baseline model test score:")
print(rg_pipe.score(X_test,y_test))


# ## Step 3 Feature Engineering

# In[ ]:


preprocess= make_column_transformer((categoric_pre,categorical),(numeric_pre,~categorical))
pipe_rfe_ridgecv=make_pipeline(preprocess,RFECV(LinearRegression(),cv=5),RidgeCV())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
mean_score=np.mean(cross_val_score(pipe_rfe_ridgecv,X_train,y_train,cv=5))

print(mean_score)


# In[ ]:


pipe_rfe_ridgecv=make_pipeline(preprocess,PolynomialFeatures(degree=1),RFECV(LinearRegression(),cv=5),RidgeCV())
mean_score=np.mean(cross_val_score(pipe_rfe_ridgecv,X_train,y_train,cv=5))
 
print(mean_score)
pipe_rfe_ridgecv.fit(X_train,y_train)
print(pipe_rfe_ridgecv.score(X_test,y_test))


# In[ ]:


select_2=make_pipeline(preprocess,SelectKBest(k=2,score_func=f_regression),RidgeCV())
mean_score=np.mean(cross_val_score(select_2,X_train,y_train,cv=5))

print(mean_score)


#     We find that use RFECV could improve the performance, but not significantly.

# ## Step 4 Any model

# In[ ]:


preprocess= make_column_transformer((categoric_pre,categorical),(numeric_pre,~categorical))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


dtr_pipe=make_pipeline(preprocess,DecisionTreeRegressor(min_samples_split=3))
dtr_pipe.fit(X_train,y_train)
scores = cross_val_score(dtr_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Decision Tree model train cv score:")
print(mean_score)
print("Decision Tree model test score:")
print(dtr_pipe.score(X_test,y_test))

rf_pipe=make_pipeline(preprocess,RandomForestRegressor(n_estimators=100))
rf_pipe.fit(X_train,y_train)
scores = cross_val_score(rf_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Random forest model train score:")
print(mean_score)
print("Random forest model test score:")
print(rf_pipe.score(X_test,y_test))

gbr_pipe=make_pipeline(preprocess,GradientBoostingRegressor())
gbr_pipe.fit(X_train,y_train)
scores = cross_val_score(gbr_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Gradient Boosting model train score:")
print(mean_score)
print("Gradient Boosting model test score:")
print(gbr_pipe.score(X_test,y_test))


#     We can see Gradient Boosting model has a better performance. But it is still not good.

# In[ ]:


X_train_tra,X_train_val,y_train_tra,y_train_val = train_test_split(X_train,y_train,test_size=0.33,random_state=42)
pipe=make_pipeline(preprocess,RandomForestRegressor())
param_grid={'randomforestregressor__n_estimators':np.arange(100,200,10),
            'randomforestregressor__max_features':np.arange(10,20,1)}
grid=GridSearchCV(pipe,param_grid=param_grid,n_jobs=-1,return_train_score=True)
scores = cross_val_score(pipe,X_train_tra,y_train_tra, cv=10)
print("mean cross valid score on test:")
print(np.mean(scores))
grid.fit(X_train_tra, y_train_tra)
best_n_estimators=grid.best_params_['randomforestregressor__n_estimators']
best_max_features=grid.best_params_['randomforestregressor__max_features']
print("best parameter:")
print(grid.best_params_)
print("score on valid set:")
print(grid.score(X_train_val, y_train_val))

pipe=make_pipeline(preprocess,RandomForestRegressor(n_estimators=best_n_estimators,max_features=best_max_features))
pipe.fit(X_train,y_train)
print("score on test set")
print(pipe.score(X_test,y_test))


#     Though, tuning paramters does not impove the model a lot.

# ## Step 5 Feature Selections

# Identify features that are important for your best model. Which features are most influential, and which features could be removed without decrease in performance? Does removing irrelevant features make your model better? (This will be discussed in the lecture on 03/04).

# In[ ]:


preprocess= make_column_transformer((categoric_pre,categorical),(numeric_pre,~categorical))
pipe=make_pipeline(preprocess,LinearRegression())
pipe.fit(X_train,y_train)
result=permutation_importance(pipe,X_train,y_train,n_repeats=10,random_state=42,n_jobs=2)
sorted_idx=result.importances_mean.argsort()

fig,ax=plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,vert=False,labels= X_test.columns[sorted_idx])
ax.set_title("Permutation Importances(test set)")
fig.tight_layout()
plt.show()


#     As we can see in the plot, the `Enthnicity` and 'age','admission location'are the most important features. Among them, age is the most important.
#     Others could be removed.
# 

# In[ ]:


#read data from csv
df=pd.read_csv("COMB.csv")
df=df[df['age']<120]
df=df.drop(['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ADMITTIME','DEATHTIME','mortality_time','DOB','AGE','INSURANCE', 'GENDER', 'FIRST_CAREUNIT'],axis=1)
df.shape
df.info()
df.target=df['mortality_withinthirtydays']
df.data=df.drop(['mortality_withinthirtydays'],axis=1)

categoricals = df.data.select_dtypes(include='object')
numerics = df.data._get_numeric_data()
df.data.columns


# In[ ]:


X=df.data
y=df.target.values

categorical = X.dtypes==object

categoric_pre= make_pipeline(SimpleImputer(strategy='constant',fill_value='missing_value'),OneHotEncoder(handle_unknown='ignore'))
numeric_pre= make_pipeline(SimpleImputer(strategy='median'),StandardScaler())


preprocess= make_column_transformer((categoric_pre,categorical),(numeric_pre,~categorical))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


dtr_pipe=make_pipeline(preprocess,DecisionTreeRegressor(min_samples_split=3))
dtr_pipe.fit(X_train,y_train)
scores = cross_val_score(dtr_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Decision Tree model train cv score:")
print(mean_score)
print("Decision Tree model test score:")
print(dtr_pipe.score(X_test,y_test))

rf_pipe=make_pipeline(preprocess,RandomForestRegressor(n_estimators=100))
rf_pipe.fit(X_train,y_train)
scores = cross_val_score(rf_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Random forest model train score:")
print(mean_score)
print("Random forest model test score:")
print(rf_pipe.score(X_test,y_test))

gbr_pipe=make_pipeline(preprocess,GradientBoostingRegressor())
gbr_pipe.fit(X_train,y_train)
scores = cross_val_score(gbr_pipe,X_train, y_train, cv=5)
mean_score=np.mean(scores)
print("Gradient Boosting model train score:")
print(mean_score)
print("Gradient Boosting model test score:")
print(gbr_pipe.score(X_test,y_test))


# In[ ]:


select_lassocv = SelectFromModel(LassoCV(), threshold=1e-5)
pipe_lassocv = make_pipeline(preprocess, select_lassocv, RidgeCV())
np.mean(cross_val_score(pipe_lassocv, X_train, y_train, cv=10))


#     Actually, removing irrelevant features do not make our model better.

# In[ ]:





# In[ ]:




