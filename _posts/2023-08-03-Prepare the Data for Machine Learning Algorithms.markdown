---
layout: post
comments: true
title:  "Prepare the Data for Machine Learning Algorithm"
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

- Take Train set and separate the labels from your datasets since we don't want to apply the same transformations to the labels and Train set
  ```Python
    #here we drop the output attribute which you will decide from your datasets
    #then we make a copy of it to `train_set_label`

    train_set= train_set.drop('`your_output_attribute name`', axis=1)
    train_set_label=strat_train_set['`your_output_attribute name`'].copy()
  ```

after doing these two things we will be working with iour new `train_set`.

### Data Cleaning
Most of machine Learning can not work with missing features, so you have to create few functions to take of them. You have three options here:
 - Get rid of the corresponding missing values in the `dataset attribute` row.
       
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

***sklean*** provide class `SimpleImputer` to help us perform the `fillnal` easily.

```python
#import SimpleImputer class
 from sklearn.import SimpleImputer

 simple_imputer = SimpleImputer(strategy = 'median)

```
Since `SimpleImputer` can only work with `numerical data`, we have to remove any attribute from our dataset that are `non numerical attributes` like `categorical data` or `text data`.

```python
 train_set_num= train_set.drop('categorical_attribute name', axis=1)
```

Now we will have to fit the `simple_imputer` instance to the training dataset using `fit()` method

```python
 simple_imputer.fit(train_set_num)
```
**SimpleIMputer** class simply is calculating the median of the numerical instance of the whole dataset and store it in `statistic_instance`
We really don't know which attribute in our datasets will have the `nall` values, so we have to apply `simple_imputer` to apply the whole datasets.

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

You can take a quick look of your datasets to see if there is any `nall` valus remaining

```python
  train_set_tr.info()
```

### Handling Text and Categorical Attributes
Previously we say that we have to drop any `categorical attribute` in our datasets because we can not compute median in the categorical datasets.

to take a look of this categorical attribute jus do
```python
 train_set_cat=train-set[['categorical_attribute name']]

 #try to view first five row
 train_set_cat.head()
```

Machine Learning does not understand the `categorical data`, so we have to convert our `categoricak data` to `numerical data`.

Using class `OrdinalEncoder`from the `sklean.preprocessing` we can accoplish that.
```python
 from sklearn.preprocessing import OrdinalEncoder
 
 #define the instance of the class 
 ordinal_encoder = OrdinalEncoder()

 #encode the categorical data by fit_transform the ordinal_encoder in our train_set_cat data
 train_set_cat_encoded = ordinal_encoder.fit_transform(train_set_cat)
```

**fit_transform** it first use `fit` to analyse `train_set_cat` and convert all the `categorical data` to `numerical data` and then `transform` do apply the learned new `numerical data` to our `train_set_cat` data.
