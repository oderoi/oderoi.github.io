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
  {% highlight ruby %}
    #from `model_selection` module inside `sklearn` import `train_test_split` function
    from sklearn.model_selection import train_test_split

    #inside the function `tran_test_split` pass your `data` then
    #define your `test_size` ratio, then pass `random_state` seed number
    #and lastly decide weither you will `shuffle` them or not.
    
    train_set, test_set = train_test_split(`datasets`, test_size=0.2, random_state=42, shuffle=True)
  {% endhighlight %}

- Take Train set and separate the labels from your datasets since we don't want to apply the same transformations to the labels and Train set
  {% highlight ruby %}
    #here we drop the output attribute which you will decide from your datasets
    #then we make a copy of it to `train_set_label`

    train_set= train_set.drop(`your_output_attribute name`, axis=1)
    train_set_label=strat_train_set[`your_output_attribute name`].copy()
  {% endhighlight %}

after doing these two things we will be working with iour new `train_set`.

### Data Cleaning]
- Most of machine Learning can not work with missing features, so you have to create few functions to take of them. You have three options here:
    - Get rid of the corresponding missing values in the `dataset attribute` row.
        {% highlight ruby %}
         train_set.dropna(subset=['attribute name'])
        {% endhighlight %}

    - Get rid of the whole `attribute` in your dataset.
        {% highlight ruby %}
         train_set.drop('attribute name', axis=1)
        {% endhighlight %}

    - Set the values to some value (zero, them mean, the median, etc.)
        {% highlight ruby %}
         median=train_set['attribute name'].median()
         train_set['attribute name'].fillna(median, inplace=True)
        {% endhighlight %}