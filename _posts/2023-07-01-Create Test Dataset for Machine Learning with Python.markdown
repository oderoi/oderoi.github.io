---
layout: post
comments: true
title:  "Create Test Dataset for Machine Learning with Python"
date:   2023-07-01 03:21:57
excerpt: "One of the very important step when performing Machine Learning or so to speak Deep Learning is to prepare your Datasets starting by Spliting them into test, train, and validation test"
categories: test set
---
## Test Set

Splitting datasets into train set and test set is one of the common steps in machine learning, perhaps now.

Splitting data theoretically is so simple, you just need to put away 20% of your datasets as test set, or even less if your dataset is too large and remaining datasets would be training set.

Let us implement this using python:
{% highlight ruby %}
#import numpy ribraly
import numpy as np
{% endhighlight %}

{% highlight ruby %}
#create a function `split_train_test`,
#that take datasets and ration of a test 
#set as input and give training and test 
#datasets as output

def split_train_test(data, test_ratio):

  #use permutation to arrange the total number 
  #of your dataset into a random order
  shuffle_index=np.random.permutation(len(data))

  test_set_size=int(test_ration*len(data))
  test_index=shuffled_index[:test_set_size]
  train_index=shuffled_index[test_set_size:]
  return data.iloc[train_index], data.iloc[test_index]
{% endhighlight %}

**So Basicaly**
- this function take in **data** and **test ratio**, then calculate **random shuffling indexes** of our **dataset** using `np.random.permutation` then find **the size of our test set** by multipying **test ratio** and **length of our dataset** then use it to find the **indexes** of **train set** and **test set** then it uses those **indexes** to select **training set** and **test set** datasets. 

{% highlight ruby %}
train_set, test_set=split_train_test(data, 0.2)
{% endhighlight %}

**Then**
- we call out function and pass in **datasets** and **test set ratio** then assign our output to **train_set** and **test_set**.

- and then let see the length our datasets, here below.

{% highlight ruby %}
print("Total size of dataset: ",len(data))

print("Train set size : ",len(train_set))

print("Test set size: ",len(test_set))

#=>Total size of dataset: 20640

#=>Train set size: 16512

#=>Test set size: 4128
{% endhighlight %}


**now**
- we can see that the **test set** is **4,128** which is actually **20%** of our total dataset which is **20,640** and our training set is **16,512** **80%** of our total datasets.

Well, that works,...
But it is not yet perfect, say if you run this program again it will break and generate new **test set** and **train set**.

One way to solve this is to add random generator’s seed:

{% highlight ruby %}
def split_train_test(data, test_ratio, random_seed):
  rseed=np.random.RandomState(random_seed)
  shuffle_index=rseed.permutation(len(data))
  test_set_size=int(test_ration*len(data))
  test_index=shuffled_index[:test_set_size]
  train_index=shuffled_index[test_set_size:]
  return data.iloc[train_index], data.iloc[test_index]
{% endhighlight %}

**So Basicaly**
- As before we create a function that take in **data** , **test ratio** and **random seed** which could be any number, then create the instance of a `Numpy` class **RandoState** with specified **random seed** then **random shuffling** the total amount of our **dataset** using **permutation** then find **the size of our test set** by multipying **test ratio** and **length of our dataset** then use it to find the **indexes** of **train set** and **test set** then it uses those **indexes** to select **training set** and **test set** datasets. 

{% highlight ruby %}
train_set, test_set=split_train_test(data, 0.2, 42)
{% endhighlight %}

**Then**
- we call again out function and pass in **datasets** , **test set ratio** and **random seed** then assign our output to **train_set** and **test_set**.

It worth noting that, this above function can be done simply by using sklearn.

Here is the implementation:

{% highlight ruby %}
from sklearn.model_selection import train_test_split
{% endhighlight %}

{% highlight ruby %}
train_set, test_set=train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
{% endhighlight %}

**So Basicaly**
- This function takes in **data** , **test size ratio** , **random state** and you can choose if you want to **shuffle** your **datasets** or not bu assigning **True** or **False** to **shuffle** then it give out **train set** and **test set** datasets. 

**Now** By adding **random seed** our program will not break when we run this code again, but when I fetch an update dataset, my code will breaks as well and generate new **test set** and **train sets**.

### So what is the solution then
One way to solve this is by using identifier of each data point or instance to decide whether or not it should go to test or train set.

We could compute hash of each instance’s identifier, and then add instances in tests set if the hash is less or equal to 20% of the maximum hash value.

This help to ensure that test and train set remain consistency even if you refresh datasets and across multiple runs.

The new test set will contain 20% of the new instance and will not contain any instance that was previous in the train set.

Here is the possible implementation.

{% highlight ruby %}
from zlib import crc32
{% endhighlight %}

{% highlight ruby %}
#computing instance's identifier hash
#and test them, if there rae less or equal to 20% of maximum hash

def test_set_check(id, test_ration):
  return crc(no.int62(id)) & 0xffffffff < test_ration ^ 2 ** 32
{% endhighlight %}

{% highlight ruby %}
#spliting instances using instance identifier hash\

def train_test_set_with_id(data, test_ration, id):
ids=data[id]
test_set_id=ids.apply(lambda _id:test_set_check(_id, test_ratio))
  return data.loc[~test_set_id], data.loc[test_set_id]
{% endhighlight %}

{% highlight ruby %}
train_set_with_id, test_set_with_id=train_test_set_with_id(data, test_size=0.2, 'id')
{% endhighlight %}

Unfortunately, some datasets have no identifiers.

The possible solution here is to create row identifier:

{% highlight ruby %}
data_with_id=data.reset_index()
{% endhighlight %}

This add a row identifier called “index” in your datasets, which is good but you have to take the followings into consideration:

- Any added data it has to be added at the end of your dataset.

- No row has to be deleted ever.

If these are not possible for you to follow, then you can create a unique identifier.

The simplest way to create a unique identifier is by taking the attributes with the instances that will never change (constant).

You may combine two attributes or more or even take a single attribute and assign it as the new attribute.

E.g.

{% highlight ruby %}
data_with_id['id']=data[attribute1] + data[attribute2]
{% endhighlight %}

It worth Noting that all the above implementations we have done so far is the random sampling.

Now it is fine to perform Random Sampling if your datasets are large enough, but if you have small datasets, you will be running the risk of introducing a significant Sampling Bias.

So, if you have small datasets the go to way is to perform Stratified sampling.

In Stratified Sampling the datasets are divided into homogeneous subgroups called Strata.

Now using Strata, the right number instances will be sampled from each Stratum to guarantee that the test set is representative of overall dataset.

Good thing is, Spliting our datasets using Stratified Sampling can be done easirly using sklearn class `StratifiedShuffleSplit` cross-validator.

**Ok**
- So this cross-validator object is the  combination of StratifiedKFold and ShuffleSplit, which returns Stratified randomized folds. The folds are made by preserving the percentage of samples for each class.

**Not**
- This methode is for sizeable / small datasets.

So here is the emplementation

{% highlight ruby %}
#from model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for index, (train_index, test_index) in enumerate(split.split(data, data[important_categorical_attribute])):
    start_train_set=data.loc[train_index]

    trat_test_set=data.loc[test_index]
{% endhighlight %}

**Ok**
- Now will this work for every scinario in our datasets, sadly **No**.

In our `for loop` remember we use `data[important_categorical_attribute]`.
**So what do we mean by important_categorical_attribute**
- In our datasets we have to look for the attribute that is very important to predict the output, hence `data[important_categorical_attribute]`

-But that attribute it need to be **categorical attribute**.

Now what if in our datasets there is no important categorical attribute.

In this case, we will have to create **categorical attribute** from very important attribute to predict the output.

**So here is how will we do it**
- So first is to identify how is your **continues sttribute** clustered.
- Here you may use normal observation by visualizing your attribute.

{% highlight ruby %}
    data[continues_attribute].hist()
{% endhighlight %}

So from the histogram you may decide how many clusters is your data
Eg: 
- you may see maybe your **continues_attribute** is clustered around 1.5 to 6 maybe so it's like category 1 is `0. - 1.5`, category 2 is `1.5 - 3.0`, category 3 is `3.0 - 4.5`, category 4 is `4.5 - 6.0` and category 5 is `6.0 - `

**Now lt's create our categorical attribute**
We will use a function `cut` from `pandas` to segment and sort data values into `bins`, so by doing that we will be able to convert continous attribute to categorical attribute.

{% highlight ruby %}
#first import pandas as pd
import pandas as pd

#then use pd.cut to segment the continous data into `bins` and then give each categories a label
data[categorical_attribute] = pd.cut(data[continues_attribute], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

#then you can visualize your categorical data using `hist`
data[categorical_attribute].hist()
{% endhighlight %}

**Now** that our `continous attribute` is converted into `categorical attirbute` we can use this attribute into our sklearn class `StratifiedShuffleSplit` `for loop` above.

**Also** with the test set genarated with random sampling (test_set) and StratifiedSampling (strat_test_set), if we compare the income category of test_set, strat_test_set and overall dataste we supporse to see that test_set set generate with Stratified Sampling proportions almost identicle to those of full dataset, while test set generate with Random Sampling is bit skewed.

**Let see then**

{% highlight ruby %}
strat_train_set[categorical_attribute].value_counts()/ len(strat_train_set)
{% endhighlight %}

{% highlight ruby %}
data[categorical_attribute].value_counts()/ len(strat_train_set)
{% endhighlight %}

{% highlight ruby %}
from sklearn.models import train_test_split
train,test=train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train[categorical_attribute].value_counts()/len(test)
{% endhighlight %}

**Last but not Least** we will have to drop `categorical_attribute` from our Stratified Sampling train and test datasets so that our data will return to it's original **attirbutes**.

{% highlight ruby %}
#here we will drop `attr` from `start_train_set` and `strat_test_set`
attr=[categorical_attribute]

for _set in (strat_train_set, strat_test_set):
    _set.drop(attr, axis=1, inplace=True)
{% endhighlight %}

**Now** our datasets will be ready for **training**