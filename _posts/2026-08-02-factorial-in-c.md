---
layout: post
title: "Factorial in C"
date: 2026-08-02 10:00:00 +0300
mathjax: true
---

Ever wonder how many ways you can arrange a deck of cards? Or how many different passwords you can make? That's where {% include term.html name="factorial" text="factorial" %} comes in — one of the most fundamental ideas in both math and programming.

## What is Factorial?

The {% include term.html name="factorial" text="factorial" %} of a {% include term.html name="non_negative_integer" text="non-negative integer" %} $n$, written as $n!$, is the {% include term.html name="product" text="product" %} of all positive integers from $1$ up to $n$.

$$n! = n \times (n-1) \times (n-2) \times \cdots \times 3 \times 2 \times 1$$

A neat recursive way to think about it:

$$n! = n \times (n-1)!$$

With the **base case**: $0! = 1$ and $1! = 1$.

### Why does $0! = 1$?

It might seem weird, but think of it this way: there's exactly **one way** to arrange zero items — do nothing! Mathematically, it also keeps the recursive formula $n! = n \times (n-1)!$ working for all $n \ge 1$.

### Examples

| $n$ | $n!$ | Calculation |
|-----|------|-------------|
| 0 | 1 | By definition |
| 1 | 1 | $1$ |
| 3 | 6 | $3 \times 2 \times 1$ |
| 5 | 120 | $5 \times 4 \times 3 \times 2 \times 1$ |
| 10 | 3,628,800 | $10 \times 9 \times \cdots \times 1$ |

Notice how fast factorial grows? $20!$ is already **2.4 quintillion** — that's why we use `double` in C to handle large values.

## The Code

Here's a clean, recursive implementation in C:

{% highlight c %}
#include &lt;stdio.h&gt;

// Function declaration
double factorial(double n);

int main() {
    double n = 20;

    // Call the factorial function
    double result = factorial(n);

    printf("%.0f! = %.0f\n", n, result);
    return 0;
}

// Recursive factorial: n! = n * (n-1)!
double factorial(double n) {
    // Base case: stops the recursion
    if (n &lt;= 1) {
        return 1;
    }

    // Recursive step: n! = n * (n-1)!
    return n * factorial(n - 1);
}
{% endhighlight %}

## How Recursion Works Here

When you call `factorial(5)`, here's what happens on the {% include term.html name="call_stack" text="call stack" %}:
factorial(5) returns 5 * factorial(4)
factorial(4) returns 4 * factorial(3)
factorial(3) returns 3 * factorial(2)
factorial(2) returns 2 * factorial(1)
factorial(1) returns 1          ← base case!


Then the results bubble back up: $1 \times 2 \times 3 \times 4 \times 5 = 120$.

### ⚠️ Watch Out for Stack Overflow!

If you call `factorial(100000)`, you'll hit a {% include term.html name="stack_overflow" text="stack overflow" %} — the {% include term.html name="call_stack" text="call stack" %} has limited space. For very large numbers, use an {% include term.html name="iteration" text="iterative" %} approach with a {% include term.html name="for_loop" text="for loop" %} instead.

## Iterative Version (No Recursion)

{% highlight c %}
#include <stdio.h>

int main() {
    int n = 5;
    double result = 1;

    // Iterative approach using a for loop
    for (int i = 1; i <= n; i++) {
        result *= i;
    }

    printf("%d! = %.0f\n", n, result);
    return 0;
}
{% endhighlight %}

Both versions give the same result, but the iterative one uses constant memory — no risk of {% include term.html name="stack_overflow" text="stack overflow" %}!

## Real-World Uses

- **Combinatorics**: How many ways to arrange $n$ items? $n!$
- **Probability**: Calculating permutations and combinations
- **Taylor series**: Approximating functions like $e^x$, $\sin(x)$, $\cos(x)$
- **Computer graphics**: Shading algorithms use factorials

## Try It Yourself

1. Modify the code to compute $10!$, $15!$, and $20!$
2. What happens if you input a negative number? Add error handling!
3. Rewrite it using a {% include term.html name="for_loop" text="while loop" %} instead of {% include term.html name="recursion" text="recursion" %}