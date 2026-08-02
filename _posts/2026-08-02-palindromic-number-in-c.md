---
layout: post
title: "Palindromic Number in C"
date: 2026-08-02 03:49:00 +0300
mathjax: true
---

{% include term.html name="palindrome" text="Palindromic number" %} (also known as a {% include term.html name="palindrome" text="numeral palindrome" %} or a {% include term.html name="palindrome" text="numeric palindrome" %}) is a number (such as **16461**) that remains the same when its digits are reversed.

In other words it has {% include term.html name="reflectional_symmetry" text="reflectional symmetry" %} across a vertical axis.

The term palindromic is derived from palindrome, which refers to a word (such as **rotor**, **tenet**) whose spelling is unchanged when its letters are reversed.

Consider a number $n > 0$ in base $b \ge 2$ where it is written in standard notation with $k + 1$ digits $a_{i}$ as:

$$n = \sum_{i=0}^{k} a_{i} b^{i}$$

where: $0 \le a_{i} < b$ for all $i$ and $a_{k} \ne 0$

Then $n$ is palindromic if and only if $a_{i} = a_{k - i}$ for all $i$. **Zero** is written $0$ in any base and is also palindromic by definition.

Note: Although palindromic numbers are most often considered in the {% include term.html name="decimal" text="decimal" %} system, the concept of **palindromicity** can be applied to the {% include term.html name="natural_numbers" text="natural numbers" %} in any {% include term.html name="numeral_system" text="numeral system" %}.

Example of 20 first palindromic numbers (in decimal):

$$0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, \cdots$$

The following code in **C** will check if the given number by the user is a palindromic number or not.

{% highlight c %}
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
{% endhighlight %}