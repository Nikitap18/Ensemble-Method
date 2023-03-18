```python
# import lib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

```


```python
df=pd.read_csv("FraudCheck.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Undergrad</th>
      <th>Marital_Status</th>
      <th>Taxable_Income</th>
      <th>City_Population</th>
      <th>Work_Experience</th>
      <th>Urban</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NO</td>
      <td>Single</td>
      <td>68833</td>
      <td>50047</td>
      <td>10</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YES</td>
      <td>Divorced</td>
      <td>33700</td>
      <td>134075</td>
      <td>18</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NO</td>
      <td>Married</td>
      <td>36925</td>
      <td>160205</td>
      <td>30</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YES</td>
      <td>Single</td>
      <td>50190</td>
      <td>193264</td>
      <td>15</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NO</td>
      <td>Married</td>
      <td>81002</td>
      <td>27533</td>
      <td>28</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>YES</td>
      <td>Divorced</td>
      <td>76340</td>
      <td>39492</td>
      <td>7</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>596</th>
      <td>YES</td>
      <td>Divorced</td>
      <td>69967</td>
      <td>55369</td>
      <td>2</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>597</th>
      <td>NO</td>
      <td>Divorced</td>
      <td>47334</td>
      <td>154058</td>
      <td>0</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>598</th>
      <td>YES</td>
      <td>Married</td>
      <td>98592</td>
      <td>180083</td>
      <td>17</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>599</th>
      <td>NO</td>
      <td>Divorced</td>
      <td>96519</td>
      <td>158137</td>
      <td>16</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 6 columns</p>
</div>



For Using Bagging, Random Forest , Stacking & XGBoost to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" and others are "Good"..we split the column values 'taxable_income' into two sets as Good and Risky


```python
df.loc[df["Taxable_Income"]>=30000,"taxable_income"]="Good"
df.loc[df["Taxable_Income"]<=30000,"taxable_income"]="Risky"

df.drop(["Taxable_Income"],axis=1,inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Undergrad</th>
      <th>Marital_Status</th>
      <th>City_Population</th>
      <th>Work_Experience</th>
      <th>Urban</th>
      <th>taxable_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NO</td>
      <td>Single</td>
      <td>50047</td>
      <td>10</td>
      <td>YES</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YES</td>
      <td>Divorced</td>
      <td>134075</td>
      <td>18</td>
      <td>YES</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NO</td>
      <td>Married</td>
      <td>160205</td>
      <td>30</td>
      <td>YES</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YES</td>
      <td>Single</td>
      <td>193264</td>
      <td>15</td>
      <td>YES</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NO</td>
      <td>Married</td>
      <td>27533</td>
      <td>28</td>
      <td>NO</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>




```python
label_encoder = preprocessing.LabelEncoder()
df['taxable_income']= label_encoder.fit_transform(df['taxable_income'])  
```


```python
df1=pd.get_dummies(df)
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City_Population</th>
      <th>Work_Experience</th>
      <th>taxable_income</th>
      <th>Undergrad_NO</th>
      <th>Undergrad_YES</th>
      <th>Marital_Status_Divorced</th>
      <th>Marital_Status_Married</th>
      <th>Marital_Status_Single</th>
      <th>Urban_NO</th>
      <th>Urban_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50047</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>134075</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>160205</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>193264</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27533</td>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>39492</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>596</th>
      <td>55369</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>597</th>
      <td>154058</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>598</th>
      <td>180083</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>599</th>
      <td>158137</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 10 columns</p>
</div>




```python
df1.columns
```




    Index(['City_Population', 'Work_Experience', 'taxable_income', 'Undergrad_NO',
           'Undergrad_YES', 'Marital_Status_Divorced', 'Marital_Status_Married',
           'Marital_Status_Single', 'Urban_NO', 'Urban_YES'],
          dtype='object')




```python
data1=list(df1.columns)
data1
```




    ['City_Population',
     'Work_Experience',
     'taxable_income',
     'Undergrad_NO',
     'Undergrad_YES',
     'Marital_Status_Divorced',
     'Marital_Status_Married',
     'Marital_Status_Single',
     'Urban_NO',
     'Urban_YES']




```python
# reorder columns
order=['taxable_income','City_Population','Work_Experience','Undergrad_NO','Undergrad_YES','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single','Urban_NO','Urban_YES']
df1=df1[order]
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxable_income</th>
      <th>City_Population</th>
      <th>Work_Experience</th>
      <th>Undergrad_NO</th>
      <th>Undergrad_YES</th>
      <th>Marital_Status_Divorced</th>
      <th>Marital_Status_Married</th>
      <th>Marital_Status_Single</th>
      <th>Urban_NO</th>
      <th>Urban_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50047</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>134075</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>160205</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>193264</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>27533</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>0</td>
      <td>39492</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>596</th>
      <td>0</td>
      <td>55369</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>597</th>
      <td>0</td>
      <td>154058</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>598</th>
      <td>0</td>
      <td>180083</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>599</th>
      <td>0</td>
      <td>158137</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 10 columns</p>
</div>




```python
#To normalize the data using normalisation
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
```


```python
#We consider the set of independent variables for normalization.
data=norm_func(df1.iloc[:,0:7])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxable_income</th>
      <th>City_Population</th>
      <th>Work_Experience</th>
      <th>Undergrad_NO</th>
      <th>Undergrad_YES</th>
      <th>Marital_Status_Divorced</th>
      <th>Marital_Status_Married</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.139472</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.622394</td>
      <td>0.600000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.772568</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.962563</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.010081</td>
      <td>0.933333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>0.0</td>
      <td>0.078811</td>
      <td>0.233333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>596</th>
      <td>0.0</td>
      <td>0.170058</td>
      <td>0.066667</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>597</th>
      <td>0.0</td>
      <td>0.737240</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>598</th>
      <td>0.0</td>
      <td>0.886810</td>
      <td>0.566667</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>599</th>
      <td>0.0</td>
      <td>0.760683</td>
      <td>0.533333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 7 columns</p>
</div>




```python
array = df1.values
X = array[:,1:]
Y = array[:,0]
seed = 154

```


```python
# Bagged Decision Trees for Classification
kfold = KFold(n_splits=15, random_state=seed,shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model,X,Y,cv=kfold)
print(results)
#print(results.mean())
```

    [0.75  0.775 0.825 0.75  0.7   0.65  0.75  0.725 0.675 0.85  0.675 0.75
     0.75  0.775 0.675]
    


```python
print(results.mean())
```

    0.7383333333333334
    


```python
results
```




    array([0.75 , 0.775, 0.825, 0.75 , 0.7  , 0.65 , 0.75 , 0.725, 0.675,
           0.85 , 0.675, 0.75 , 0.75 , 0.775, 0.675])




```python
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
```


```python
X = array[:,1:]
Y = array[:,0]

```


```python
num_trees = 100
max_features = 3
kfold = KFold(n_splits=15, random_state=7,shuffle=True)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model,X,Y,cv=kfold)
print(results)
```

    [0.725 0.8   0.575 0.85  0.775 0.825 0.725 0.75  0.725 0.75  0.75  0.55
     0.825 0.75  0.75 ]
    


```python
results.mean()
```




    0.7416666666666667




```python
# Adaboost
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
```


```python
X = array[:,1:]
Y = array[:,0]
```


```python
num_trees = 100
seed=154
kfold = KFold(n_splits=15, random_state=seed,shuffle=True)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
```

    [0.825 0.775 0.85  0.8   0.75  0.7   0.8   0.7   0.725 0.85  0.8   0.825
     0.825 0.825 0.75 ]
    


```python
results.mean()
```




    0.7866666666666665




```python
# Stacking Ensemble for Classification
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
```


```python
X = array[:,1:]
Y = array[:,0]
kfold = KFold(n_splits=15, random_state=7,shuffle=True)

```


```python
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results)
```

    [0.8   0.8   0.675 0.9   0.85  0.925 0.725 0.825 0.725 0.8   0.825 0.625
     0.85  0.775 0.8  ]
    


```python
results.mean()
```




    0.7933333333333333




```python
# First XGBoost model 

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

```


```python
# split data into X and y
X = array[:,1:]
Y = array[:,0]

```


```python
array
```




    array([[     0,  50047,     10, ...,      1,      0,      1],
           [     0, 134075,     18, ...,      0,      0,      1],
           [     0, 160205,     30, ...,      0,      0,      1],
           ...,
           [     0, 154058,      0, ...,      0,      0,      1],
           [     0, 180083,     17, ...,      0,      1,      0],
           [     0, 158137,     16, ...,      0,      1,      0]], dtype=int64)




```python
# split data into train and test sets
seed = 154
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

```


```python
# fit model no training data
model = XGBClassifier(n_estimators=100,max_depth=3)
model.fit(X_train, y_train)

```




    XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric=None, feature_types=None,
                  gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
                  interaction_constraints=None, learning_rate=None, max_bin=None,
                  max_cat_threshold=None, max_cat_to_onehot=None,
                  max_delta_step=None, max_depth=3, max_leaves=None,
                  min_child_weight=None, missing=nan, monotone_constraints=None,
                  n_estimators=100, n_jobs=None, num_parallel_tree=None,
                  predictor=None, random_state=None, ...)




```python
# make predictions for test data
y_pred = model.predict(X_test)
y_pred
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
predictions = [round(value) for value in y_pred]
predictions
```




    [0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0]




```python
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.4f%%" % (accuracy * 100.0))
```

    Accuracy: 81.6667%
    

Inference:  Model on fraud data treating those who have taxable_income <= 30000 having                             Bagging accuracy= 73.8%,
            Adaboost= 78.66%,
            XGBoost=  81.67%
            Stacking=  79.33%
            RondomForest= 74.16%
            
         
