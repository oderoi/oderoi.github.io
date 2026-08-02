---
layout: post
title: "Binary Search in C — Finding Needles in Haystacks"
date: 2026-08-02 12:00:00 +0300
excerpt: "Master binary search with visual intuition, C implementations, and real-world applications"
categories: [C, Algorithms]
mathjax: true
---

Imagine you're looking for a word in a dictionary. Do you start at page 1 and flip through every page? Of course not! You open somewhere in the middle, decide if your word is before or after, and repeat. That's exactly how {% include term.html name="binary_search" text="binary search" %} works.

## Why Binary Search?

If you have a {% include term.html name="sorted_array" text="sorted array" %} of 1 million numbers, a {% include term.html name="sequential_search" text="sequential search" %} (checking one by one) could take up to 1 million steps. {% include term.html name="binary_search" text="Binary search" %}? Just **20 steps**. That's the power of $O(\log n)$ {% include term.html name="time_complexity" text="time complexity" %}.

## The Intuition

1. Find the {% include term.html name="midpoint" text="midpoint" %} of the {% include term.html name="array" text="array" %}
2. Is the {% include term.html name="target_value" text="target" %} at the midpoint? Done!
3. Is the target smaller? Search the left half
4. Is the target larger? Search the right half
5. Repeat until found (or the range is empty)

![Binary Search Depiction](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Binary_Search_Depiction.svg/500px-Binary_Search_Depiction.svg.png)

*Image: {% include term.html name="binary_search" text="Binary search" %} repeatedly halves the search space. [Wikipedia](https://en.wikipedia.org/wiki/Binary_search_algorithm)*

## The Math Behind It

If you halve $n$ repeatedly, how many times until you reach 1?

$$\log_2(n) = \text{number of halvings}$$

For $n = 1,000,000$:
$$\log_2(1,000,000) \approx 19.93$$

So {% include term.html name="binary_search" text="binary search" %} needs at most **20 comparisons** for a million elements!

## The Algorithm

Given a {% include term.html name="sorted_array" text="sorted array" %} $X$ with $n$ elements and a {% include term.html name="target_value" text="target value" %} $T$:

1. Set $S = 0$ (start {% include term.html name="index" text="index" %})
2. Set $E = n - 1$ (end {% include term.html name="index" text="index" %})
3. While $E \ge S$:
   - $m = S + \frac{E - S}{2}$ (avoid overflow with this formula!)
   - If $X_m = T$: return $m$ (found!)
   - If $X_m &gt; T$: set $E = m - 1$ (search left)
   - If $X_m &lt; T$: set $S = m + 1$ (search right)
4. Return $-1$ (not found)

![Binary Search](https://upload.wikimedia.org/wikipedia/commons/c/c1/Binary-search-work.gif)

## Visual Walkthrough

Let's trace the algorithm searching for **21** in this {% include term.html name="array" text="array" %}:

![Binary Search Step-by-Step](/assets/images/binary_search_steps.png)

*Each step eliminates half the remaining elements. Yellow = midpoint checked, Green = target found.*

Here is the same trace in table form:

| Step | `start` | `end` | `mid` | `arr[mid]` | Comparison | Action |
|:----:|:-------:|:-----:|:-----:|:----------:|:----------:|:------:|
| 1 | 0 | 16 | 8 | 14 | 14 &lt; 21 | Search right: `start = 9` |
| 2 | 9 | 16 | 12 | 24 | 24 &gt; 21 | Search left: `end = 11` |
| 3 | 9 | 11 | 10 | 19 | 19 &lt; 21 | Search right: `start = 11` |
| 4 | 11 | 11 | 11 | 21 | 21 == 21 | **Found at index 11!** |

**Only 4 comparisons** to find the target in 17 elements. A linear search would have taken 12 comparisons.

## Why This Works

Binary search relies on a powerful invariant: **if the array is sorted, every element left of `mid` is smaller, and every element right of `mid` is larger.**

This lets us safely discard half the search space after each comparison. The key insight is that we never "lose" the target — we only eliminate regions where the target *cannot* exist.

The overflow-safe midpoint formula `start + (end - start) / 2` is critical. The naive `(start + end) / 2` can overflow when `start` and `end` are large integers near `INT_MAX`.

## The C Code

Here is the iterative implementation — the version you should use in production:

```c
#include <stdio.h>

// Binary search function
// Returns the index of target if found, -1 otherwise
int binary_search(int arr[], int start, int end, int target);

int main() {
    // A sorted array (binary search REQUIRES sorted data!)
    int arr[] = {1, 3, 4, 6, 7, 8, 10, 13, 14, 18, 19, 21, 24, 37, 40, 45, 71};
    int n = sizeof(arr) / sizeof(arr[0]);

    int target;
    printf("Enter a number to search: ");
    scanf("%d", &target);

    // Search!
    int index = binary_search(arr, 0, n - 1, target);

    if (index == -1) {
        printf("❌ %d was not found in the array.\n", target);
    } else {
        printf("✅ %d found at index %d!\n", target, index);
    }

    return 0;
}

int binary_search(int arr[], int start, int end, int target) {
    // Keep searching while the range is valid
    while (start &lt;= end) {
        // Calculate midpoint (this formula prevents integer overflow!)
        int mid = start + (end - start) / 2;

        // Check if target is at midpoint
        if (arr[mid] == target) {
            return mid;  // Found it!
        }

        // If target is smaller, search left half
        if (arr[mid] &gt; target) {
            end = mid - 1;
        }
        // If target is larger, search right half
        else {
            start = mid + 1;
        }
    }

    // Target not found
    return -1;
}
```

## Expected Output

Enter a number to search: 21
✅ 21 found at index 11!
Enter a number to search: 99
❌ 99 was not found in the array.


## Recursive Version

{% include term.html name="binary_search" text="Binary search" %} is a classic {% include term.html name="divide_and_conquer" text="divide and conquer" %} problem. Here's the recursive version:

```c
#include <stdio.h>

int binary_search_recursive(int arr[], int start, int end, int target) {
    // Base case: target not found
    if (start > end) {
        return -1;
    }

    int mid = start + (end - start) / 2;

    // Base case: found the target!
    if (arr[mid] == target) {
        return mid;
    }

    // Divide and conquer: search the appropriate half
    if (arr[mid] > target) {
        return binary_search_recursive(arr, start, mid - 1, target);
    } else {
        return binary_search_recursive(arr, mid + 1, end, target);
    }
}

int main() {
    int arr[] = {2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 23;

    int result = binary_search_recursive(arr, 0, n - 1, target);

    if (result != -1) {
        printf("Element found at index %d\n", result);
    } else {
        printf("Element not found\n");
    }

    return 0;
}
```

## Iterative vs Recursive

| Approach | Time | Space | Pros | Cons |
|:--------:|:----:|:-----:|------|------|
| **Iterative** | $O(\log n)$ | $O(1)$ | No {% include term.html name="stack_overflow" text="stack overflow" %} risk, faster, constant memory | Slightly more code to read |
| **Recursive** | $O(\log n)$ | $O(\log n)$ | Clean, elegant, matches math definition | Risk of {% include term.html name="stack_overflow" text="stack overflow" %} on huge arrays |

For production code, use the **iterative** version. For learning and interviews, both are valuable.

## Time Complexity Comparison

| Algorithm | Best Case | Average Case | Worst Case | Space | Requires Sorted? |
|:---------:|:---------:|:------------:|:----------:|:-----:|:----------------:|
| {% include term.html name="sequential_search" text="Linear Search" %} | $O(1)$ | $O(n)$ | $O(n)$ | $O(1)$ | No |
| {% include term.html name="binary_search" text="Binary Search" %} | $O(1)$ | $O(\log n)$ | $O(\log n)$ | $O(1)$ | **Yes** |

## Common Pitfalls

1. **Integer overflow**: Use `mid = start + (end - start) / 2` **NOT** `(start + end) / 2`. When `start` and `end` are both near `INT_MAX`, their sum overflows.
2. **Off-by-one errors**: Be careful with `mid - 1` vs `mid`. If you set `end = mid` instead of `end = mid - 1`, you can get an infinite loop.
3. **Unsorted input**: {% include term.html name="binary_search" text="Binary search" %} only works on {% include term.html name="sorted_array" text="sorted data" %}! Running it on unsorted data gives unpredictable (and wrong) results.
4. **Infinite loops**: Make sure `start` or `end` changes every {% include term.html name="iteration" text="iteration" %}. If neither changes, the loop never exits.

## Real-World Uses

- **Database indexing**: Finding records in B-trees and B+ trees — the backbone of MySQL, PostgreSQL, and MongoDB
- **Git bisect**: Finding which commit introduced a bug by binary searching through commit history
- **Auto-complete**: Finding prefix matches in sorted dictionaries and search suggestions
- **Version control**: Finding the first bad version in a release history (LeetCode classic)
- **Graphics**: Ray tracing acceleration structures use spatial binary search (BVH trees)
- **Compilers**: Symbol table lookups in sorted identifier lists

## Try It Yourself

1. **Missing element**: What happens if you search for a number not in the array? Trace through the algorithm manually with `target = 100`.
2. **First occurrence**: Modify the code to find the **first** occurrence of a duplicate value. (Hint: when `arr[mid] == target`, don't return immediately — keep searching left.)
3. **Square root**: Implement {% include term.html name="binary_search" text="binary search" %} to find the integer square root of a number. Search for $x$ where $x^2 \le n < (x+1)^2$.
4. **Performance test**: Compare the speed of {% include term.html name="sequential_search" text="linear search" %} vs {% include term.html name="binary_search" text="binary search" %} on an {% include term.html name="array" text="array" %} of 10 million elements. Time both with `clock()`.
5. **Rotated array**: What if the sorted array was rotated at some pivot? Can you still use binary search? (LeetCode #33)