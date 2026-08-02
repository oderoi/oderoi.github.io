---
layout: post
title: "Matrix Multiplication in C — A Visual Guide"
date: 2024-07-16
excerpt: "Learn matrix multiplication from the ground up with C code, visual intuition, and real-world applications"
categories: [C, Math]
mathjax: true
---

Matrices aren't just grids of numbers — they're the language of {% include term.html name="linear_algebra" text="linear algebra" %}, powering everything from video games to AI neural networks. Let's learn {% include term.html name="matrix_multiplication" text="matrix multiplication" %} from the ground up.

## What is a Matrix?

A matrix is a rectangular {% include term.html name="array" text="array" %} of numbers arranged in rows and columns. An $m \times n$ matrix has $m$ rows and $n$ columns.

$$A = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix} \quad \text{(a 4×4 matrix)}$$

Each {% include term.html name="element" text="element" %} is identified by its row and column {% include term.html name="index" text="indices" %}: $A_{i,j}$ is the element at row $i$, column $j$.

## The Dot Product

Before multiplying matrices, you need to understand the {% include term.html name="dot_product" text="dot product" %}.

Given two {% include term.html name="vector" text="vectors" %} of the same length, the {% include term.html name="dot_product" text="dot product" %} multiplies corresponding entries and sums them:

$$\vec{a} \cdot \vec{b} = \sum_{k=0}^{n-1} a_k \cdot b_k = a_0b_0 + a_1b_1 + \cdots + a_{n-1}b_{n-1}$$

For example:
$$\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} = 1(1) + 2(1) + 3(1) + 4(1) = 10$$

## Matrix Multiplication Rule

To multiply $A \times B$:

1. **Check dimensions**: $A$ must be $m \times k$ and $B$ must be $k \times n$. The **inner dimensions** ($k$) must match!
2. **Result**: The product $C = A \times B$ will be $m \times n$.
3. **Each element**: $C_{i,j}$ = {% include term.html name="dot_product" text="dot product" %} of row $i$ of $A$ and column $j$ of $B$.

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \times B_{k,j}$$

### Visual Intuition

![Matrix Multiplication Visual](/assets/images/matrix_mult_visual.png)

*Top: Each cell in the result is the dot product of a row from the first matrix and a column from the second. Bottom: The triple nested loop that implements this.*

## Why This Works

Matrix multiplication is essentially a **system of linear equations**. Each row of $A$ defines a linear combination, and each column of $B$ provides the coefficients. The result $C$ encodes how these transformations compose.

The dimension rule ($A_{cols} = B_{rows}$) exists because the {% include term.html name="dot_product" text="dot product" %} requires equal-length {% include term.html name="vector" text="vectors" %}. You can't dot a 4-element row with a 3-element column.

## The C Code

Here is a clean implementation with proper memory management:

```c
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;

/*
 * Matrix Multiplication in C
 * A[rowA][colA] × B[rowB][colB] = C[rowA][colB]
 * 
 * Rule: colA MUST equal rowB for multiplication to be valid.
 */

#define ROW_A 4
#define COL_A 4
#define ROW_B 4
#define COL_B 4

// Function to multiply two matrices
// Returns a pointer to the resulting matrix C
int** multiply_matrices(int A[][COL_A], int B[][COL_B]);

int main() {
    // Matrix A: 4 rows, 4 columns
    int A[ROW_A][COL_A] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    // Matrix B: 4 rows, 4 columns
    // Note: COL_A (4) must equal ROW_B (4) ✓
    int B[ROW_B][COL_B] = {
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4}
    };

    printf("Matrix A (%dx%d):\n", ROW_A, COL_A);
    for (int i = 0; i &lt; ROW_A; i++) {
        for (int j = 0; j &lt; COL_A; j++) {
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B (%dx%d):\n", ROW_B, COL_B);
    for (int i = 0; i &lt; ROW_B; i++) {
        for (int j = 0; j &lt; COL_B; j++) {
            printf("%d\t", B[i][j]);
        }
        printf("\n");
    }

    // Multiply!
    int **C = multiply_matrices(A, B);

    printf("\nResult C = A × B (%dx%d):\n", ROW_A, COL_B);
    for (int i = 0; i &lt; ROW_A; i++) {
        for (int j = 0; j &lt; COL_B; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }

    // CRITICAL: Free allocated memory to prevent memory leaks
    for (int i = 0; i &lt; ROW_A; i++) {
        free(C[i]);
    }
    free(C);

    return 0;
}

int** multiply_matrices(int A[][COL_A], int B[][COL_B]) {
    // Allocate result matrix C[ROW_A][COL_B] on the heap
    int **C = (int**)malloc(ROW_A * sizeof(int*));
    for (int i = 0; i &lt; ROW_A; i++) {
        C[i] = (int*)malloc(COL_B * sizeof(int));
    }

    // Triple nested loop for matrix multiplication
    for (int i = 0; i &lt; ROW_A; i++) {          // rows of A
        for (int j = 0; j &lt; COL_B; j++) {      // columns of B
            C[i][j] = 0;                        // initialize cell

            // Compute dot product of row i of A and column j of B
            for (int k = 0; k &lt; COL_A; k++) {  // COL_A == ROW_B
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}
```

## Expected Output

Matrix A (4x4):
1	2	3	4
5	6	7	8
9	10	11	12
13	14	15	16
Matrix B (4x4):
1	2	3	4
1	2	3	4
1	2	3	4
1	2	3	4
Result C = A × B (4x4):
10	20	30	40
26	52	78	104
42	84	126	168
58	116	174	232


## Step-by-Step Verification

Let's verify one cell manually: $C_{0,0}$

$$C_{0,0} = A_{0,0} \cdot B_{0,0} + A_{0,1} \cdot B_{1,0} + A_{0,2} \cdot B_{2,0} + A_{0,3} \cdot B_{3,0}$$

$$C_{0,0} = (1 \times 1) + (2 \times 1) + (3 \times 1) + (4 \times 1) = 1 + 2 + 3 + 4 = 10 \checkmark$$

And $C_{1,2}$:

$$C_{1,2} = (5 \times 3) + (6 \times 3) + (7 \times 3) + (8 \times 3) = 15 + 18 + 21 + 24 = 78 \checkmark$$

## Why Use Pointers?

The function returns `int**` (a {% include term.html name="pointer" text="pointer" %} to {% include term.html name="pointer" text="pointers" %}) because C needs to allocate the result matrix dynamically on the {% include term.html name="heap" text="heap" %}. The {% include term.html name="call_stack" text="stack" %} can't easily return 2D arrays.

**Always free your memory!** Forgetting `free()` causes a {% include term.html name="memory_leak" text="memory leak" %} — your program gradually consumes more RAM until it crashes or slows down.

## Common Mistakes

| Mistake | Why It Breaks | How to Fix |
|:--------|:--------------|:-----------|
| $A_{cols} \neq B_{rows}$ | Dimensions must match for multiplication | Verify `COL_A == ROW_B` before calling |
| Forgetting to `malloc` | {% include term.html name="pointer" text="Dangling pointer" %} → crash | Always allocate before use |
| Not freeing memory | {% include term.html name="memory_leak" text="Memory leak" %} | `free()` each row, then the array of pointers |
| Wrong loop order | Wrong result or out-of-bounds access | Follow `i`→`j`→`k` pattern |
| Integer overflow | Large matrix values exceed `int` range | Use `long long` for big matrices |

## Identity Matrix Verification

The {% include term.html name="identity_matrix" text="identity matrix" %} $I$ is the \"1\" of matrix multiplication.

$$A \times I = A$$

```c
// 4x4 Identity Matrix
int I[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
};
// Multiply A × I and verify you get A back
```

## Time Complexity

| Operation | Complexity | Explanation |
|:----------|:-----------|:------------|
| **Matrix Multiplication** | $O(m \cdot k \cdot n)$ | Three nested loops: rows × inner dim × cols |
| **Square matrices ($n \times n$)** | $O(n^3)$ | Classic cubic complexity |
| **Space** | $O(m \cdot n)$ | Result matrix storage |

For large matrices, $O(n^3)$ is too slow. Real-world systems use:
- **Strassen's algorithm**: $O(n^{2.81})$
- **Coppersmith-Winograd**: $O(n^{2.37})$ (theoretical)
- **GPU parallelization**: Process thousands of cells simultaneously

## Real-World Applications

| Domain | How Matrix Multiplication is Used |
|:-------|:----------------------------------|
| **Computer Graphics** | 3D transformations: rotation, scaling, translation of objects |
| **AI / Machine Learning** | Neural networks are essentially layers of matrix operations |
| **PageRank** | Google's original algorithm uses matrix {% include term.html name="iteration" text="iteration" %} |
| **Image Processing** | Filters like blur and edge detection use matrix convolution |
| **Physics Simulations** | Solving systems of linear equations in engineering |
| **Economics** | Input-output models (Leontief matrix) |

## Try It Yourself

1. **Identity check**: Multiply any $4 \times 4$ matrix by the {% include term.html name="identity_matrix" text="identity matrix" %}. What do you notice? (Hint: $A \times I = A$)
2. **Dimension mismatch**: What happens if you set `COL_A = 3` and `ROW_B = 4`? Does it compile? Does it run correctly?
3. **Cache optimization**: Can you optimize the code by transposing matrix $B$ first? (Hint: better {% include term.html name="cache" text="cache" %} locality!)
4. **Big matrices**: Change the code to use `double` instead of `int` and multiply two $100 \times 100$ random matrices. Time it with `clock()`.
5. **Memory detective**: Comment out the `free()` calls, run the program in a loop 100,000 times, and watch your system's memory usage with `top` or Task Manager.