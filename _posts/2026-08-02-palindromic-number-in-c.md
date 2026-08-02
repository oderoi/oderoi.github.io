---
layout: post
title: "Palindromic Number in C — The Mirror Test"
date: 2026-08-02 03:49:00 +0300
excerpt: "Learn to detect palindromic numbers with digit manipulation, visual intuition, and C code"
categories: [C, Math]
mathjax: true
---

A {% include term.html name="palindrome" text="palindromic number" %} is a number that reads the same forwards and backwards — like **16461**, **12321**, or **7**. It has {% include term.html name="reflectional_symmetry" text="reflectional symmetry" %} across a vertical axis: the first {% include term.html name="digit" text="digit" %} matches the last, the second matches the second-to-last, and so on.

The concept extends beyond numbers. A word like **{% include term.html name="rotor" text="rotor" %}** or **tenet** is also a palindrome. In fact, the term comes from the Greek *palíndromos* — "running back again."

## The Mirror Test

The simplest way to check if a number is palindromic? **Reverse its digits and compare.**

![Palindrome Visual](/assets/images/palindrome_visual.png)

*Top: The digit-reversal algorithm extracts each digit, builds the reverse, and compares. Bottom: A palindrome mirrors perfectly; a non-palindrome breaks the symmetry.*

## Why This Works

Our {% include term.html name="algorithm" text="algorithm" %} relies on two elementary operations:

1. **`number % 10`** — the {% include term.html name="modulo" text="modulo" %} operator extracts the **last digit**
2. **`number / 10`** — integer division **chops off** the last digit

By repeating these two operations in a {% include term.html name="while_loop" text="while loop" %}, we peel off digits one by one (like layers of an onion) and build the reversed number:

| Operation | `number` | `number % 10` | `number / 10` |
|:---------:|:--------:|:-------------:|:-------------:|
| Start | 16461 | 1 | 1646 |
| After 1st | 1646 | 6 | 164 |
| After 2nd | 164 | 4 | 16 |
| After 3rd | 16 | 6 | 1 |
| After 4th | 1 | 1 | 0 |

When `number` becomes 0, we've extracted every digit. The reversed number is built by `reverse = reverse × 10 + digit` at each step.

## Step-by-Step Trace

Let's trace `num = 16461` through the algorithm:

| Iteration | `number` | `rem = number % 10` | `reverse = reverse×10 + rem` | `number = number / 10` |
|:---------:|:--------:|:-------------------:|:----------------------------:|:----------------------:|
| 1 | 16461 | 1 | 0×10 + 1 = **1** | 1646 |
| 2 | 1646 | 6 | 1×10 + 6 = **16** | 164 |
| 3 | 164 | 4 | 16×10 + 4 = **164** | 16 |
| 4 | 16 | 6 | 164×10 + 6 = **1646** | 1 |
| 5 | 1 | 1 | 1646×10 + 1 = **16461** | 0 |

**Final check:** `origin (16461) == reverse (16461)` → ✅ **Palindrome!**

Now trace `num = 12345`:

| Iteration | `number` | `rem` | `reverse` | `number / 10` |
|:---------:|:--------:|:-----:|:---------:|:-------------:|
| 1 | 12345 | 5 | 5 | 1234 |
| 2 | 1234 | 4 | 54 | 123 |
| ... | ... | ... | ... | ... |
| 5 | 1 | 1 | 54321 | 0 |

**Final check:** `origin (12345) == reverse (54321)` → ❌ **Not a palindrome!**

## The C Code

Here is a clean implementation with proper error handling:

```c
#include <stdio.h>

// Returns 1 if palindrome, 0 otherwise
int is_palindrome(int number);

int main() {
    int num;

    printf("Enter a number to check: ");
    scanf("%d", &num);

    // Error handling: negative numbers
    if (num < 0) {
        printf("Negative numbers are not considered palindromic in this implementation.\n");
        return 1;
    }

    if (is_palindrome(num)) {
        printf("✅ %d is a palindromic number!\n", num);
    } else {
        printf("❌ %d is NOT a palindromic number.\n", num);
    }

    return 0;
}

int is_palindrome(int number) {
    // Save the original number for comparison
    int original = number;
    int reverse = 0;

    // Reverse the number digit by digit
    while (number != 0) {
        int remainder = number % 10;        // Extract last digit
        reverse = reverse * 10 + remainder; // Append to reverse
        number = number / 10;               // Remove last digit
    }

    // Palindrome if reversed equals original
    return (original == reverse);
}
```

## Expected Output

Enter a number to check: 16461
✅ 16461 is a palindromic number!
Enter a number to check: 12345
❌ 12345 is NOT a palindromic number.
Enter a number to check: 7
✅ 7 is a palindromic number!


## How the Reversal Works

The key insight is place-value arithmetic. Each time we do `reverse = reverse × 10 + digit`, we **shift all existing digits left by one place** (the `× 10`) and **insert the new digit at the ones place**.

reverse = 0
reverse = 0×10 + 1 = 1          // digit: 1
reverse = 1×10 + 6 = 16         // digit: 6
reverse = 16×10 + 4 = 164       // digit: 4
reverse = 164×10 + 6 = 1646     // digit: 6
reverse = 1646×10 + 1 = 16461   // digit: 1


This is exactly how you'd reverse a number by hand — just automated.

## Common Pitfalls

| Mistake | Why It Breaks | Fix |
|:--------|:--------------|:----|
| Forgetting to save `original` | You compare `reverse` with `number`, but `number` is now 0 | Store `original = number` before the loop |
| Using `number % 10` on negatives | Negative remainders in C (-123 % 10 = -3) | Check `number < 0` first or use `abs()` |
| Integer overflow | `reverse * 10` can exceed `INT_MAX` for large inputs | Use `long long` for 10+ digit numbers |
| Single-digit numbers | Some forget that 0–9 are palindromes | The algorithm handles this correctly (0→0, 7→7) |

## Alternative: The Two-Pointer Approach

Instead of reversing the entire number, you can compare digits from both ends:

```c
#include <stdio.h>
#include <math.h>

int is_palindrome_two_pointer(int number) {
    if (number < 0) return 0;
    if (number < 10) return 1;  // Single digit is always palindrome

    // Count digits
    int digits = 0;
    int temp = number;
    while (temp != 0) {
        digits++;
        temp /= 10;
    }

    // Compare outer digits, then move inward
    int left = digits - 1;
    int right = 0;

    while (left > right) {
        int left_digit = (number / (int)pow(10, left)) % 10;
        int right_digit = (number / (int)pow(10, right)) % 10;

        if (left_digit != right_digit) {
            return 0;  // Mismatch found
        }

        left--;
        right++;
    }

    return 1;
}
```

The reversal method is simpler and faster for most cases. The two-pointer approach shines when you **cannot** modify or copy the number (e.g., in a read-only memory context).

## Real-World Connections

| Domain | Application |
|:-------|:------------|
| **Genetics** | Palindromic DNA sequences (e.g., restriction enzyme sites) read the same on complementary strands |
| **Number Theory** | Palindromic primes (e.g., 131, 151, 181) and their distribution properties |
| **Date Calculations** | 02/02/2020 was a palindromic date in both DD/MM/YYYY and MM/DD/YYYY formats |
| **Error Detection** | Certain checksum algorithms exploit palindromic patterns for validation |
| **Puzzles & Competitions** | Project Euler #4: largest palindrome from the product of two 3-digit numbers = 906609 |
| **Linguistics** | Palindrome detection in DNA sequencing, poetry analysis, and word games |

## Mathematical Definition (Formal)

For the curious, a number $n > 0$ in base $b \ge 2$ with digits $a_0, a_1, \ldots, a_k$:

$$n = \sum_{i=0}^{k} a_i b^i \quad \text{where } 0 \le a_i < b \text{ and } a_k \ne 0$$

Then $n$ is palindromic if and only if:

$$a_i = a_{k-i} \quad \text{for all } i \in \{0, 1, \ldots, k\}$$

**Zero** is palindromic by definition in any base.

## Try It Yourself

1. **Project Euler #4:** What is the largest palindromic number made from the product of two 3-digit numbers? (Answer: 906609 = 913 × 993)
2. **Binary palindromes:** Modify the code to check palindromes in base-2. Is 5 (`101` in binary) a palindrome? Is 6 (`110`)?
3. **No reversal:** Can you check palindromicity without reversing the number or converting it to a string? (Hint: compare the first and last digits mathematically.)
4. **Longest palindrome:** Write a program that finds the longest palindromic substring in a given string. (This is a classic interview problem!)
5. **Palindrome generator:** Write a program that generates all palindromic numbers between 1 and 10,000.
6. **Overflow hunter:** What happens if you input `2147483647` (INT_MAX)? Does the reversal overflow? Fix it by using `long long`.