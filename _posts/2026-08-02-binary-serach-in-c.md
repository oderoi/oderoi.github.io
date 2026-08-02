---
layout: post
title: "Binary Search in C — Finding Needles in Haystacks"
date: 2026-08-02 12:00:00 +0300
mathjax: true
---

Imagine you're looking for a word in a dictionary. Do you start at page 1 and flip through every page? Of course not! You open somewhere in the middle, decide if your word is before or after, and repeat. That's exactly how {% include term.html name="binary_search" text="binary search" %} works.

## Why Binary Search?

If you have a {% include term.html name="sorted_array" text="sorted array" %} of 1 million numbers, a {% include term.html name="sequential_search" text="sequential search" %} (checking one by one) could take up to 1 million steps. {% include term.html name="binary_search" text="Binary search" %}? Just **20 steps**. That's the power of $O(\log n)$ {% include term.html name="time_complexity" text="time complexity" %}.

## The Intuition

1. Find the {% include term.html name="midpoint" text="midpoint" %} of the array
2. Is the target at the midpoint? Done!
3. Is the target smaller? Search the left half
4. Is the target larger? Search the right half
5. Repeat until found (or the range is empty)

![Binary Search](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Binary_Search_Depiction.svg/500px-Binary_Search_Depiction.svg.png)

*Image: {% include term.html name="binary_search" text="Binary search" %} repeatedly halves the search space.*

## The Math Behind It

If you halve $n$ repeatedly, how many times until you reach 1?

$$\log_2(n) = \text{number of halvings}$$

For $n = 1,000,000$:
$$\log_2(1,000,000) \approx 20$$

So {% include term.html name="binary_search" text="binary search" %} needs at most 20 comparisons for a million elements!

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

## The C Code

{% highlight c %}
#include &lt;stdio.h&gt;

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
{% endhighlight %}

## Recursive Version

{% include term.html name="binary_search" text="Binary search" %} is a classic {% include term.html name="divide_and_conquer" text="divide and conquer" %} problem. Here's the recursive version:

{% highlight c %}
#include &lt;stdio.h&gt;

int binary_search_recursive(int arr[], int start, int end, int target) {
    // Base case: target not found
    if (start &gt; end) {
        return -1;
    }

    int mid = start + (end - start) / 2;

    // Base case: found the target!
    if (arr[mid] == target) {
        return mid;
    }

    // Divide and conquer: search the appropriate half
    if (arr[mid] &gt; target) {
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
{% endhighlight %}

## Code Output Example

Enter a number to search: 21
✅ 21 found at index 11!
Enter a number to search: 99
❌ 99 was not found in the array.

## Iterative vs Recursive

| Approach | Pros | Cons |
|----------|------|------|
| **Iterative** | No {% include term.html name="stack_overflow" text="stack overflow" %} risk, faster | Slightly more code |
| **Recursive** | Clean, elegant, matches math definition | Risk of {% include term.html name="stack_overflow" text="stack overflow" %} on huge arrays |

For production code, use the **iterative** version. For learning and interviews, both are valuable.

## Time Complexity Comparison

| Algorithm | Best | Average | Worst | Requires Sorted? |
|-----------|------|---------|-------|-----------------|
| {% include term.html name="sequential_search" text="Linear Search" %} | $O(1)$ | $O(n)$ | $O(n)$ | No |
| {% include term.html name="binary_search" text="Binary Search" %} | $O(1)$ | $O(\log n)$ | $O(\log n)$ | **Yes** |

## Common Pitfalls

1. **Integer overflow**: Use `mid = start + (end - start) / 2` NOT `(start + end) / 2`
2. **Off-by-one errors**: Be careful with `mid - 1` vs `mid`
3. **Unsorted input**: {% include term.html name="binary_search" text="Binary search" %} only works on {% include term.html name="sorted_array" text="sorted data" %}!
4. **Infinite loops**: Make sure `start` or `end` changes every {% include term.html name="iteration" text="iteration" %}

## Real-World Uses

- **Database indexing**: Finding records in B-trees
- **Git bisect**: Finding which commit introduced a bug
- **Auto-complete**: Finding prefix matches in sorted dictionaries
- **Version control**: Finding the first bad version in a release history

## Try It Yourself

1. What happens if you search for a number not in the array?
2. Modify the code to find the **first** occurrence of a duplicate value
3. Implement {% include term.html name="binary_search" text="binary search" %} to find the square root of a number (hint: search for $x$ where $x^2 = n$)
4. Compare the speed of {% include term.html name="sequential_search" text="linear search" %} vs {% include term.html name="binary_search" text="binary search" %} on an array of 10 million elements