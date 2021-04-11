# Missing Data Imputation 

* How to deal with missing that if 1% of the features were missing? 
  * => We could remove the records if it is only small amount
* What about if 50% of the features are missing
  * =>  It is better to impute missing data  
* Missing data impuation methods 
  * Mean/median/mode
  * Random 
  * KNN or other interpolation
  * Grid search to find the best method



# Implementation

1. Split data into train and test 

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis=1), 
                                                    df['SalePrice'], 
                                                    test_size=0.3, 
                                                    random_state=0)
```

2. Check missing data 

```python
# numerical columns vs. categorical columns 
num_cols = X_train.select_dtypes('number').columns
cat_cols = X_train.select_dtypes('object').columns

# numerical columns and categorical columns that contain missing data
num_cols_with_na = num_cols[X_train[num_cols].isna().mean() > 0]
cat_cols_with_na = cat_cols[X_train[cat_cols].isna().mean() > 0]

# percentage of missing data 
X_train[num_cols_with_na].isna().mean().sort_values(ascending=False)
X_train[cat_cols_with_na].isna().mean().sort_values(ascending=False)
```

3. Impute missing data 

```python
from sklearn.impute import SimpleImputer

# initialize imputer. 
# use strategy='median' for median imputation
# use strategy='most_frequent' for mode imputation
imputer = SimpleImputer(strategy='mean')

# fit the imputer on X_train
imputer.fit(X_train[num_cols_with_na])

# transform the data using the fitted imputer
X_train_mean_impute = imputer.transform(X_train[num_cols_with_na])

# put the output into DataFrame. remember to pass columns used in fit/transform
X_train_mean_impute = pd.DataFrame(X_train_mean_impute, columns=num_cols_with_na)
```

4. Add missing data indicator (Optional. Done in the previous step)

```python
imputer = SimpleImputer(strategy='mean', add_indicator=True)

# add extra columns when creating 
X_train_mean_impute = pd.DataFrame(X_train_mean_impute, columns=num_cols_with_na.append(num_cols_with_na + '_NA')) # <- _NA colums
```

