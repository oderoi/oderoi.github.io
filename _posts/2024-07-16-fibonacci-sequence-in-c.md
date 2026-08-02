---
layout: post
title: "Fibonacci Sequence in C"
date: 2024-07-16
excerpt: "Understanding recursion and the golden ratio through Fibonacci"
categories: [C, Math]
mathjax: true
---

The {% include term.html name="sequence" text="Fibonacci sequence" %} is a {% include term.html name="recurrence" text="recurrence relation" %} where each number is the sum of the two {% include term.html name="element" text="elements" %} that precede it.

## The Math

$$F_0 = 0,\quad F_1 = 1$$
$$F_n = F_{n-1} + F_{n-2} \quad \text{for } n > 1$$

## Why This Works (Recursion Tree)

factorial(5)
├── factorial(4)
│   ├── factorial(3)
│   │   ├── factorial(2)
│   │   │   ├── factorial(1) → 1  (base case!)
│   │   │   └── factorial(0) → 0  (base case!)

## The C Code

{% highlight c %}
#include <stdio.h>

int fibonacci(int n);

int main() {
    int n = 10;
    for (int i = 0; i < n; i++) {
        printf("%d\t", fibonacci(i));
    }
    printf("\n");
    return 0;
}

int fibonacci(int n) {
    if (n == 0) return 0;   // {% include term.html name="base_case" text="base case" %}
    if (n == 1) return 1;   // {% include term.html name="base_case" text="base case" %}
    return fibonacci(n - 1) + fibonacci(n - 2);  // {% include term.html name="recursion" text="recursive" %} step
}
{% endhighlight %}

## Expected Output

0	1	1	2	3	5	8	13	21	34

## Real-World Context

- **Nature:** Petal arrangements, pinecones, nautilus shells
- **Finance:** Technical analysis retracement levels
- **Algorithms:** Introduces memoization and dynamic programming

## Try It Yourself

1. Modify the code to print the first 20 Fibonacci numbers.
2. What happens when `n = 50`? Why is it so slow? (Hint: count the repeated calls)
3. Implement an iterative version using a {% include term.html name="for_loop" text="for loop" %}.