---
layout: post
title: "Palindromic Number in C"
date: 2026-08-02 03:49:00 +0300
mathjax: true
---

Palindromic number (also known as a _**numeral palindrome**_ or a _**numeric palindrome**_) is a number (such as _**16461**_) that remains the same when its digits are reversed.

In other words it has __**reflectional symmetry**__ across a vertical axis.

The term palindromic is derived from palindrome, which refers to a word (such as _**rotor**_, _**tenet**_) whose spelling is unchanged when its letters are reversed.

Consider a number $n > 0$ in base $b \ge 2$ where it is written in standard notation with $k + 1$ digits $a_{i}$ as:

$$n = \sum_{i=0}^{k} a_{i} b^{i}$$

where: $0 \le a_{i} < b$ for all $i$ and $a_{k} \ne 0$

Then $n$ is palindromic if and only if $a_{i} = a_{k - i}$ for all $i$. _**Zero**_ is written $0$ in any base and is also palindromic by definition.

Note: Although palindromic numbers are most often considered in the _**decimal**_ system, the concept of _**palindromicity**_ can be applied to the _**natural numbers**_ in any _**numeral system**_.

Example of 20 first palindromic numbers (in decimal):

$$0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, \cdots$$

The following code in _**C**_ will check if the given number by the user is a palindromic number or not.

```c
/*
This code helps us to look if a given number is palindromic or not.
*/

#include <stdio.h>

int palindromic_no(int number, int reverse, int remainder);

int main() {

    // Enter a number to check.
    int num;
    printf("Enter number: ");
    scanf("%d", &num);

    int rev = 0;
    int rem;

    // check is done here
    palindromic_no(num, rev, rem);
    return 0;
}

int palindromic_no(int number, int reverse, int remainder) {

    // reversing the given number
    int origin_no = number;
    while (number != 0) {
        remainder = number % 10;
        reverse = reverse * 10 + remainder;
        number /= 10;
    }

    // check if given number is palindromic number or not.
    if (origin_no == reverse) {
        printf("Number: %d, is palindromic number", origin_no);
    } else {
        printf("Number: %d, is not palindromic number", origin_no);
    }
    return 0;
}