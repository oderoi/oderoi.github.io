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
def split_train_test(data, test_ratio):
  shuffle_index=np.random.permutation(len(data))
  test_set_size=int(test_ration*len(data))
  test_index=shuffled_index[:test_set_size]
  train_index=shuffled_index[test_set_size:]
  return data.iloc[train_index], data.iloc[test_index]
{% endhighlight %}

{% highlight ruby %}
train_set, test_set=split_train_test(data, 0.2)
{% endhighlight %}

{% highlight ruby %}
len(data)
#=>20640
{% endhighlight %}

{% highlight ruby %}
len(test_set)
#=>4128
{% endhighlight %}

{% highlight ruby %}
len(train_test)
#=>16512
{% endhighlight %}

Well, this works, but it is not perfect, if you run program again it will break and generate new test and train set.

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

{% highlight ruby %}
train_set, test_set=split_train_test(data, 0.2, 42)
{% endhighlight %}

It worth noting that, this above function can be done simply by using sklearn.

Here is the implementation:

{% highlight ruby %}
from sklearn.model_selection import train_test_split
{% endhighlight %}

{% highlight ruby %}
train_set, test_set=train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
{% endhighlight %}

This solve the problem but when I fetch an update dataset, this also breaks and generate new test and train sets.

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
