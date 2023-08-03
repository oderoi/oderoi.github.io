---
layout: post
comments: true
title:  "Prepare the Data for Machine Learning Algorithm"
date:   2023-08-03 18:54
excerpt: "
In the realm of machine learning, data preparation is the hidden key that unlocks the true potential of algorithms. Just as a sculptor carefully shapes and refines raw materials to create a masterpiece, data preparation involves transforming and refining raw data into a well-structured, clean, and meaningful format. This crucial step sets the foundation for accurate and effective machine learning models, determining the very essence of success in any data-driven endeavor. In this blog post, we embark on an enlightening journey through the art of preparing data for machine learning algorithms, unraveling the techniques and best practices that empower us to unleash the full power of artificial intelligence.
"
---

So before doing anything with your datasets we first have to do two things first.
- Split your datasets into train set and test set.
    - {% highlight ruby %}
        #from `model_selection` module inside `sklearn` import `train_test_split` function
       from sklearn.model_selection import train_test_split

       #inside the function `tran_test_split` pass your `data` then
       #define your `test_size` ratio, then pass `random_state` seed number
       #and lastly decide weither you will `shuffle` them or not.
        
       train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
        {% endhighlight %}

- Take Train set and separate the labels from your datasets
    - {% highlight ruby %}
        #here we drop the output attribute which you will decide from your datasets
        #then we make a copy of it to `train_set_label`

        train_set= train_set.drop(your_output_attribute, axis=1)
        train_set_label=strat_train_set[your_output_attribute].copy()
        {% endhighlight %}

after doing these two things we will be working with iour new `train_set`.

### Data Cleaning