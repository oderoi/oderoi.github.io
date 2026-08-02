---
layout: null
title: "Matrix Multiplication in C — A Visual Guide"
date: 2026-08-02 11:00:00 +0300
mathjax: true
---

Matrices aren't just grids of numbers — they're the language of {% include term.html name="linear_algebra" text="linear algebra" %}, powering everything from video games to AI neural networks. Let's learn {% include term.html name="matrix_multiplication" text="matrix multiplication" %} from the ground up.

## What is a Matrix?

A matrix is a rectangular array of numbers arranged in rows and columns. An $m \times n$ matrix has $m$ rows and $n$ columns.

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad \text{(a 3×2 matrix)}$$

## The Dot Product

Before multiplying matrices, you need to understand the {% include term.html name="dot_product" text="dot product" %}.

Given two vectors of the same length, the {% include term.html name="dot_product" text="dot product" %} multiplies corresponding entries and sums them:

$$\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + \cdots + a_nb_n$$

For example:
$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$$

## Matrix Multiplication Rule

To multiply $A \times B$:

1. **Check dimensions**: $A$ must be $m \times k$ and $B$ must be $k \times n$. The inner dimensions ($k$) must match!
2. **Result**: The product $C = A \times B$ will be $m \times n$.
3. **Each element**: $C_{i,j}$ = {% include term.html name="dot_product" text="dot product" %} of row $i$ of $A$ and column $j$ of $B$.

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \times B_{k,j}$$

### Visual Intuition

![Matrix Multiplication](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Matrix_multiplication_diagram.svg/500px-Matrix_multiplication_diagram.svg.png)

*Image: Each cell in the result is the {% include term.html name="dot_product" text="dot product" %} of a row from the first matrix and a column from the second.*

## The C Code

```c
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;

/*
 * Matrix Multiplication in C
 * A[m][k] × B[k][n] = C[m][n]
 */

#define ROW_A 3
#define COL_A 2
#define ROW_B 2
#define COL_B 3

// Function to multiply two matrices
int** multiply_matrices(int A[][COL_A], int B[][COL_B]);

int main() {
    // Matrix A: 3 rows, 2 columns
    int A[ROW_A][COL_A] = {
        {1, 2},
        {3, 4},
        {5, 6}
    };

    // Matrix B: 2 rows, 3 columns
    // Note: COL_A (2) must equal ROW_B (2) ✓
    int B[ROW_B][COL_B] = {
        {1, 2, 3},
        {4, 5, 6}
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

    // Free allocated memory
    for (int i = 0; i &lt; ROW_A; i++) {
        free(C[i]);
    }
    free(C);

    return 0;
}

int** multiply_matrices(int A[][COL_A], int B[][COL_B]) {
    // Allocate result matrix C[ROW_A][COL_B]
    int **C = (int**)malloc(ROW_A * sizeof(int*));
    for (int i = 0; i &lt; ROW_A; i++) {
        C[i] = (int*)malloc(COL_B * sizeof(int));
    }

    // Triple nested loop for matrix multiplication
    for (int i = 0; i &lt; ROW_A; i++) {         // rows of A
        for (int j = 0; j &lt; COL_B; j++) {     // columns of B
            C[i][j] = 0;                       // initialize cell

            // Compute dot product of row i of A and column j of B
            for (int k = 0; k &lt; COL_A; k++) {  // COL_A == ROW_B
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}
```

## Output
Matrix A (3x2):
1	2
3	4
5	6
Matrix B (2x3):
1	2	3
4	5	6
Result C = A × B (3x3):
9	12	15
19	26	33
29	40	51


Let's verify one cell: $C_{0,0} = (1 \times 1) + (2 \times 4) = 1 + 8 = 9$ ✓

## Why Use Pointers?

The function returns `int**` (a {% include term.html name="pointer" text="pointer" %} to {% include term.html name="pointer" text="pointers" %}) because C needs to allocate the result matrix dynamically on the heap. The {% include term.html name="call_stack" text="stack" %} can't easily return 2D arrays.

## Common Mistakes

| Mistake | Why It Breaks |
|---------|--------------|
| $A_{cols} \neq B_{rows}$ | Dimensions must match for multiplication |
| Forgetting to `malloc` | {% include term.html name="pointer" text="Dangling pointer" %} → crash |
| Not freeing memory | Memory leak |
| Wrong loop order | Wrong result or out-of-bounds access |

## Identity Matrix Verification

The {% include term.html name="identity_matrix" text="identity matrix" %} $I$ is the \"1\" of matrix multiplication.

$$A \times I = A$$

{% highlight c %}
// 4x4 Identity Matrix
int I[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
};
// Multiply A × I and verify you get A back
{% endhighlight %}

## Real-World Applications

- **Graphics**: Transforming 3D coordinates (rotation, scaling, translation)
- **AI/ML**: Neural networks are essentially layers of matrix operations
- **PageRank**: Google's original algorithm uses matrix {% include term.html name="iteration" text="iteration" %}
- **Image processing**: Filters like blur and edge detection use matrix convolution

## Challenge

1. Modify the code to multiply a $4 \times 4$ identity matrix with any $4 \times 4$ matrix. What do you notice?
2. Can you optimize the code by transposing matrix $B$ first? (Hint: better cache performance!)
3. What happens if you multiply by the {% include term.html name="identity_matrix" text="identity matrix" %}?