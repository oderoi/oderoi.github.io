---
layout: post
title: "Factorial in C — From Counting to Infinity"
date: 2026-08-02 10:00:00 +0300
excerpt: "Master factorial with recursion, iteration, and visual intuition in C"
categories: [C, Math]
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
|:---:|:-----:|:------------|
| 0 | 1 | By definition |
| 1 | 1 | $1$ |
| 3 | 6 | $3 \times 2 \times 1$ |
| 5 | 120 | $5 \times 4 \times 3 \times 2 \times 1$ |
| 10 | 3,628,800 | $10 \times 9 \times \cdots \times 1$ |
| 15 | 1,307,674,368,000 | $15!$ |
| 20 | 2,432,902,008,176,640,000 | $20!$ |

Notice how fast factorial grows? $20!$ is already **2.4 quintillion** — that's why we need `double` or `long long` in C to handle large values.

## Visual Intuition

![Factorial Growth and Call Stack](/assets/images/factorial_post_visual.png)

*Top: Factorial grows faster than any exponential function. It exceeds 32-bit `int` at $n=13$ and 64-bit `long long` at $n=21$. Bottom: The call stack pushes frames down to the base case, then pops them back up, multiplying as it unwinds.*

## Why This Works

Factorial is the simplest example of **recursion with a clear base case**:

1. **The problem shrinks:** $n!$ depends on $(n-1)!$, which is a smaller version of the same problem.
2. **The base case stops it:** When $n \le 1$, we return 1 immediately. Without this, the function would call itself forever.
3. **The stack unwinds:** Each frame waits for the one below it to return, then multiplies the result by its own $n$.

This pattern — **divide, recurse, combine** — appears in merge sort, quicksort, tree traversals, and countless other algorithms.

## The Recursive C Code

Here's a clean recursive implementation:

```c
#include <stdio.h>

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
```

## Expected Output

20! = 2432902008176640000

## How Recursion Works Here

When you call `factorial(5)`, here's what happens on the {% include term.html name="call_stack" text="call stack" %}:

PUSH (winding down):
main() calls factorial(5)
factorial(5) calls factorial(4)
factorial(4) calls factorial(3)
factorial(3) calls factorial(2)
factorial(2) calls factorial(1)
factorial(1) → returns 1  ✓ BASE CASE!
POP (unwinding back up):
factorial(2) returns 2 × 1 = 2
factorial(3) returns 3 × 2 = 6
factorial(4) returns 4 × 6 = 24
factorial(5) returns 5 × 24 = 120


Then the final result bubbles back to `main()`: **120**.

### ⚠️ Watch Out for Stack Overflow!

If you call `factorial(100000)`, you'll hit a {% include term.html name="stack_overflow" text="stack overflow" %} — the {% include term.html name="call_stack" text="call stack" %} has limited space (typically ~1–8 MB). Each recursive call pushes a new frame, and 100,000 frames will crash your program.

For very large numbers, use an {% include term.html name="iteration" text="iterative" %} approach instead.

## Iterative Version (No Recursion)

```c
#include <stdio.h>

int main() {
    int n = 20;
    unsigned long long result = 1;

    // Iterative approach using a for loop
    for (int i = 1; i <= n; i++) {
        result *= i;
    }

    printf("%d! = %llu\n", n, result);
    return 0;
}
```

**Output:**

20! = 2432902008176640000


Both versions give the same result, but the iterative one:
- Uses **constant memory** — no call stack growth
- Runs in **$O(n)$ time** with **$O(1)$ space**
- Can handle much larger $n$ before overflowing

## Data Type Limits

| Type | Max Value | Highest $n!$ it can hold |
|:-----|:----------|:-------------------------|
| `int` (32-bit) | 2,147,483,647 | $12! = 479,001,600$ |
| `unsigned int` | 4,294,967,295 | $12!$ |
| `long long` (64-bit) | 9,223,372,036,854,775,807 | $20!$ |
| `unsigned long long` | 18,446,744,073,709,551,615 | $20!$ |
| `double` | ~$1.8 \times 10^{308}$ | $170!$ (loses integer precision after $15!$) |

> **Pro tip:** Use `unsigned long long` for exact integer results up to $20!$. Use `double` only when you need larger approximate values and don't care about perfect precision.

## Real-World Uses

| Domain | Application |
|:-------|:------------|
| **Combinatorics** | How many ways to arrange $n$ items? $n!$ {% include term.html name="permutation" text="permutations" %} |
| **Probability** | Calculating {% include term.html name="combination" text="combinations" %} $C(n,k) = \frac{n!}{k!(n-k)!}$ |
| **Taylor Series** | Approximating $e^x$, $\sin(x)$, $\cos(x)$ using factorial denominators |
| **Computer Graphics** | Shading algorithms and Bernstein polynomials |
| **Cryptography** | Counting possible keys and password spaces |
| **Statistical Mechanics** | Entropy calculations in physics |

### The Taylor Series Connection

Factorials appear in the denominators of Taylor series, letting us approximate transcendental functions with polynomials:

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$$

This is how calculators compute these functions — they sum the first 10–20 terms!

## Try It Yourself

1. **Scale test:** Modify the code to compute $10!$, $15!$, and $20!$. Verify the values match the table above.
2. **Error handling:** What happens if you input a negative number? Add a check: `if (n < 0) { printf("Factorial undefined for negatives!\n"); return -1; }`
3. **While loop:** Rewrite the iterative version using a `while` loop instead of a `for` loop.
4. **Type detective:** Change `double` to `int` and compute `13!`. What happens? Why? (Hint: integer overflow)
5. **Taylor approximation:** Write a program that computes $e^1$ using the Taylor series $e^x = \sum \frac{x^n}{n!}$. Sum the first 15 terms and compare with `exp(1.0)` from `<math.h>`.
6. **Permutation calculator:** Write a program that asks for $n$ and $k$, then computes $P(n,k) = \frac{n!}{(n-k)!}$ — the number of ways to arrange $k$ items from a set of $n$.