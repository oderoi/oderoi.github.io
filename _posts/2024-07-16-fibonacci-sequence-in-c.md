---
layout: post
title: "Fibonacci Sequence in C — From Nature to Code"
date: 2024-07-16
excerpt: "Understanding recursion, base cases, and the golden ratio through C programming"
categories: [C, Math]
mathjax: true
---

The {% include term.html name="sequence" text="Fibonacci sequence" %} appears everywhere — in the spiral of a nautilus shell, the arrangement of sunflower seeds, and even the branching of trees. In programming, it is the classic gateway to understanding {% include term.html name="recursion" text="recursion" %}, {% include term.html name="base_case" text="base cases" %}, and why naive solutions can hide exponential costs.

## The Mathematical Definition

The Fibonacci numbers are defined by a {% include term.html name="recurrence" text="recurrence relation" %}:

$$F_0 = 0, \quad F_1 = 1$$

$$F_n = F_{n-1} + F_{n-2} \quad \text{for } n > 1$$

Each {% include term.html name="element" text="element" %} is the sum of the two preceding elements. Simple to state — but surprisingly deep in its consequences.

## Visual Intuition: The Recursion Tree

When you call `fibonacci(5)`, the {% include term.html name="call_stack" text="call stack" %} builds a tree of function calls. Watch how the same sub-problems are solved over and over again:

                        fib(5)
                    /      \
                fib(4)        fib(3)
                /      \       /      \
            fib(3)    fib(2) fib(2)   fib(1)
            /      \   /   \   /   \
        fib(2)  fib(1) fib(1) fib(0) fib(1) fib(0)
        /    \
    fib(1)  fib(0)

| Color                           | Meaning                                       |
|---------------------------------|-----------------------------------------------|
| 🟢 `fib(1)`, `fib(0)`           | **Base cases** — these stop the recursion     |
| 🟡 `fib(2)`, `fib(3)`, `fib(4)` | **Recursive calls** — computed multiple times!|

Notice that `fib(3)` is calculated **twice**, and `fib(2)` is calculated **three times**. This redundant work is why the naive recursive approach becomes unbearably slow as $n$ grows.

![Fibonacci Recursion Tree and Growth](/assets/images/fibonacci-visualization.png)
*Left: The recursion tree for `fibonacci(5)` shows redundant calls. Right: Fibonacci growth closely tracks the golden ratio $\phi^n / \sqrt{5}$.*

## Why This Works

A recursive function needs three things to succeed:

1. **A problem that breaks into smaller versions of itself**  
   Finding $F_n$ requires finding $F_{n-1}$ and $F_{n-2}$ — smaller Fibonacci numbers.

2. **Base cases that stop the recursion**  
   Without `if (n == 0) return 0;` and `if (n == 1) return 1;`, the function would call itself forever, causing a {% include term.html name="stack_overflow" text="stack overflow" %}.

3. **A guarantee of progress toward the base cases**  
   Each recursive call reduces $n$, so we eventually reach 0 or 1.

## The C Code

Here is the classic recursive implementation:

```c
#include <stdio.h>

// Function declaration
int fibonacci(int n);

int main() {
    int n = 10;

    printf("First %d Fibonacci numbers:\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d\t", fibonacci(i));
    }
    printf("\n");

    return 0;
}

// Recursive Fibonacci: F(n) = F(n-1) + F(n-2)
int fibonacci(int n) {
    // Base cases: stop the recursion
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return 1;
    }

    // Recursive step: break into smaller problems
    return fibonacci(n - 1) + fibonacci(n - 2);
}
```

## Expected Output

First 10 Fibonacci numbers:
0	1	1	2	3	5	8	13	21	34


## The Hidden Cost: Why Recursion Is Slow

The recursive solution looks elegant, but its time complexity is $O(2^n)$. This is because each call spawns two more calls, leading to an explosion of redundant work.

| $n$ | Recursive Calls | Iterative Steps | Recursive Time |
|-----|-----------------|-----------------|----------------|
| 5   | 15              | 5               | ~0.001 ms      |
| 10  | 177             | 10              | ~0.01 ms       |
| 20  | 21,891          | 20              | ~10 ms         |
| 30  | 2,692,537       | 30              | ~1 second      |
| 40  | 331,160,281     | 40              | ~2 minutes     |
| 50  | ~40 billion     | 50              | ~2.5 years*    |

*\*Extrapolated. In practice, your program would crash from a stack overflow long before finishing.*

![Recursive vs Iterative Performance](/assets/images/fibonacci-performance.png)
*Left: The recursive approach explodes exponentially while iterative stays linear. Right: The ratio $F(n)/F(n-1)$ converges to the golden ratio $\phi \approx 1.618$.*

## Iterative Version (No Recursion)

For any real-world use, use a {% include term.html name="for_loop" text="for loop" %} instead. It runs in $O(n)$ time with $O(1)$ memory:

```c
#include <stdio.h>

int main() {
    int n = 10;

    printf("First %d Fibonacci numbers (iterative):\n", n);

    long long prev2 = 0;  // F(0)
    long long prev1 = 1;  // F(1)

    printf("%lld\t%lld\t", prev2, prev1);

    for (int i = 2; i < n; i++) {
        long long current = prev1 + prev2;  // F(i) = F(i-1) + F(i-2)
        printf("%lld\t", current);

        // Shift the window forward
        prev2 = prev1;
        prev1 = current;
    }
    printf("\n");

    return 0;
}
```

**Output:**

First 10 Fibonacci numbers:
0	1	1	2	3	5	8	13	21	34


Both versions produce the same result, but the iterative one:
- Uses constant memory (no call stack growth)
- Runs in linear time $O(n)$ instead of exponential $O(2^n)$
- Can handle $n = 10,000$ without breaking a sweat

## The Golden Ratio Connection

As $n$ grows large, the ratio of consecutive Fibonacci numbers approaches the **golden ratio**:

$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887...$$

This gives us **Binet's closed-form formula** — no recursion needed:

$$F_n = \frac{\phi^n - (1-\phi)^n}{\sqrt{5}}$$

In C, you could compute this directly with `<math.h>`, though floating-point precision limits its accuracy for large $n$.

## Real-World Connections

| Domain               | How Fibonacci Appears|
|----------------------|----------------------|
| **Nature**           | Petal counts, pinecone spirals, nautilus shells, branching patterns |
| **Finance**          | Fibonacci retracement levels in technical stock analysis |
| **Algorithms**       | Introduces memoization, dynamic programming, and recurrence relations |
| **Art & Design**     | Golden ratio proportions in architecture and visual composition |
| **Computer Science** | Analysis of recursive algorithms and the {% include term.html name="call_stack" text="call stack" %} |

## Try It Yourself

1. **Scale test:** Modify the code to print the first 20 Fibonacci numbers. At what value does a 32-bit `int` overflow? (Hint: check $F_{47}$)
2. **Performance test:** Time the recursive version for `n = 40` and `n = 45`. Why does it get so much slower?
3. **Iterative rewrite:** Convert the recursive function to use a {% include term.html name="for_loop" text="for loop" %} with an {% include term.html name="array" text="array" %} to store values.
4. **Memoization bonus:** Use an {% include term.html name="array" text="array" %} to cache previously computed values. Compare the speedup against the naive recursive version for `n = 40`.
5. **Golden ratio:** Compute $F_{30} / F_{29}$ in your calculator. How close is it to $\phi$?