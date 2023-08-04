---
layout: post
comments: true
title:  "The Art of Data Preparation for Machine Learning Algorithm"
date:   2023-08-03 18:00:57
excerpt: "In the realm of machine learning, data preparation is the hidden key that unlocks the true potential of algorithms. Just as a sculptor carefully shapes and refines raw materials to create a masterpiece, data preparation involves transforming and refining raw data into a well-structured, clean, and meaningful format. This crucial step sets the foundation for accurate and effective machine learning models, determining the very essence of success in any data-driven endeavor. In this blog post, we embark on an enlightening journey through the art of preparing data for machine learning algorithms, unraveling the techniques and best practices that empower us to unleash the full power of artificial intelligence."
categories: Unlocking the full potential of Machine Learning, Mastering the Art of Data Preparation
---

So before doing anything with your datasets we first have to do two things first.
- Split your datasets into train set and test set.

 ```python
    #from `model_selection` module inside `sklearn` import `train_test_split` function
    from sklearn.model_selection import train_test_split

    #inside the function `tran_test_split` pass your `data` then
    #define your `test_size` ratio, then pass `random_state` seed number
    #and lastly decide weither you will `shuffle` them or not.
    
    train_set, test_set = train_test_split(`datasets`, test_size=0.2, random_state=42, shuffle=True)
  ```

- Take Train set and separate the labels from your datasets since we don't want to apply the same transformations to the labels and Train set.

  ```Python
    #here we drop the output attribute which you will decide from your datasets
    #then we make a copy of it to `train_set_label`

    train_set= train_set.drop('your_output_attribute name', axis=1)
    train_set_label=strat_train_set['your_output_attribute name'].copy()
  ```

after doing these two things we will be working with our new `train_set`.

### Data Cleaning
Most of machine Learning can not work with missing features, so you have to create few functions to take care of them. You have three options here:
 - Get rid of the corresponding missing values in the dataset attribute row.
       
    ```Python
    train_set.dropna(subset=['attribute name'])
    ``` 
 - Get rid of the whole `attribute` in your dataset.

    ```Python
    train_set.drop('attribute name', axis=1)
    ```
 - Set the values to some value (zero, them mean, the median, etc.)

    ```Python
        median=train_set['attribute name'].median()
        train_set['attribute name'].fillna(median, inplace=True)
    ```

The median computed must be saved so that it can be used latter in `test set` and even new datasets to fill all the nall values.

**sklean** provide class `SimpleImputer` to help us perform the `fillnal` easily.

```python
#import SimpleImputer class
 from sklearn.import SimpleImputer

 simple_imputer = SimpleImputer(strategy = 'median')

```
Since `SimpleImputer` can only work with numerical data, we have to remove any attribute from our dataset that are non numerical attributes like categorical data or text data.

```python
 train_set_num= train_set.drop('categorical_attribute name', axis=1)
```

Now we will have to fit the `simple_imputer` instance to the training dataset using `fit()` method

```python
 simple_imputer.fit(train_set_num)
```
**SimpleIMputer** class simply is calculating the median of the numerical instance of the whole dataset and store it in `statistic_instance`
We really don't know which attribute in our datasets will have the nall values, so we have to apply `simple_imputer` to apply the whole datasets.

```python
#to see your median values just call `statistic_` instance variable
simple_imputer.statistics_
```
Now you can use the trained imputer `simple_imputer` to transform the training set by replacing missing values ny the learned medians.

```python
  X=simple_imputer.transform(train_set_num)
```

The result of the transformed feature is the plain `Numpy Array`, to convert it back to `Pandas` use `Pandas DataFrame` considering our datasets were on the `Pandas DataFrame`.

```python
import pandas as pd
 train_set_tr=pd.DataFrame(X, columns=train_set_num.columns)
```

You can take a quick look of your datasets to see if there is any nall values remaining

```python
  train_set_tr.info()
```

### Handling Text and Categorical Attributes
Previously we say that we have to drop any categorical attribute in our datasets because we can not compute median in the categorical datasets.

to take a look of this categorical attribute jus do
```python
 train_set_cat=train-set[['categorical_attribute name']]

 #try to view first five row
 train_set_cat.head()
```

Machine Learning does not understand the categorical data, so we have to convert our categoricak data to numerical data.

Using class `OrdinalEncoder`from the `sklean.preprocessing` we can accoplish that.
```python
 from sklearn.preprocessing import OrdinalEncoder
 
 #define the instance of the class 
 ordinal_encoder = OrdinalEncoder()

 #encode the categorical data by fit_transform the ordinal_encoder in our train_set_cat data
 train_set_cat_encoded = ordinal_encoder.fit_transform(train_set_cat)

 #to see the first five rows of encodes data
 train_set_cat_encoded[:5]

 #you can take a look of the categories using
 ordinal_encoder.categories_
```

**fit_transform** it first use `fit` to analyse train_set_cat and convert all the categorical data to numerical data and then `transform` do apply the learned new numerical data to our train_set_cat data.

**OrdinalEncoder** class do perform the task but one challanges with this is thatMachine Learning will assume that two nearby values are more similar than two distant values, This may be fine for few cases like for ordered categiries such as 'bad', 'average', 'good', 'excellent', but if un ordered categories that are not much related. To fix this issue you have to use `OneHotEncoder` which convert only one attribute will be equal to 1(hot), while the others will be 0(cold)y.

```python
 from sklearn.preprocessing import OneHotEncoder

 #define the instance of the class
 onehot_encoder= OneHotEncoder()

 #encode the categorical data by fit_transform the ordinal_encoder in our train_set_cat data
 train_set_cat_encoded = onehot_encoder.fit_transform(train_set_cat)
 
 #you can take a look of the categories using
onehot_encoder.categories_
```
`train_set_cat_encoded` is in Scipy sparse matrix instead of numpy array, tis help to preserve the memory ny saving just the location of the no-zero element to save the momery that would have used to store zeros, so you have to convert it to a numpy array.

```python 
 train_set_cat_encoded.toarray()
```

### Feature Scalling
Machine Learning Algorithm with few exception do not perform well when numeric attributes have different data scales, So one of the very important feature transformation you need to add to your data is **feature scaling**, there are two way to do feature scalling,
    - min-max scaling
    - standardization

**Min-max scalling** shift and rescale the values in the range of 0 to 1, you can do this by substracting min value and dividing by the max minus the min.*Scikit-Learn* provide a transformer called `MinMaxScaler` which consist of `feature_range` hyperparameter that let you change if you don't want 0 - 1 for some reasons.

**Standardization** have quit different approach, first it substract mean to make the standardize value always have zero mean, then divide with standard devition so that the standardize value distribution have unit variance, unlike Mini-Max scaling standardization do not bound values to specific range which can be a problem to Neural network where they expert input value range from 0 to 1, however Standardization is much less affected by outliers.

**sklearn**provides a transformer called `StandardScaler` and `MinMaxScaler`.

### Transformation Pipeline

As you have seen there are many transformations steps that need to exceuted in right order, ***Scikit-Learn*** provide the **Pipeline** class to help combine these different transformations process for numerical attributes.

```python
 from sklearn.preprocessing import StandardScaler, MiniMaxScaler
 from sklearn.impute import SimpleImputer

 numeric_pipeline= Pipeline([
    ('imputer'), SimpleImputer(strategy='median'),
    ('standard_scaler', StandardScaller()),
 ])
```
**Remember** We said `SimpleImputer` only work with numerical data and we have remove any categorical attribute from our dataset.

```python
 train_set_num= train_set.drop('categorical_attribute name', axis=1)

 train_set_tranform = numeric_pipeline.fit_transform(train_set_num)
```

So far we have handled categorical columns and numeriacl columns separately, It would have been more convinient to have the single transformer to handle all collumns and applying appropiate transformation to each column, Scikit-Learn introduce ***ColumnTransformer*** class for that purpose and it can work with Pandas Dataframe.
```python
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import StandardScaler, OneHotEncoder
 from sklearn.impute import SimpleImputer 

 numeric_attributes = list(train_set_num)
 categorical_attributes = ['categorical_attribute name']

 full_pipeline = ColumnTransfromer([
    ('numerical', numeric_pipeline, numeric_attributes),
    ('categorical', OneHotEncoder(), categorical_attributes)
 ])

 train_set_prepared = full_pipeline.fit_tarnform(train_set)
```

The ***train_set_prepared*** is a numpy array read to be used to train Machine Learning Algorithm

```python
 #you can view first 5 row of our datasets
 train_set_prepared[:5]
```

***Now*** our datasets will be ready for ***training*** Machine Learning Algorithm