---
layout: post
title: "Matrix multiplication in C"
date: 2024-07-16
excerpt: "Matrix multiplication using C programming"
categories: "Math Algorithms using C"
mathjax: true
---

Given two matrices A of dimensions M x K and B of dimension K x N, we want to compute their {% include term.html name="dot_product" text="dot product" %} C = A . B, which is also known as {% include term.html name="matrix_multiplication" text="matrix multiplication" %}.

The dot product.

$C += A . B$

$1D Array.$

$$C_{i} = \sum\limits_{i = 0}^{n} (A_{i} . B_{i}) = A_{1}B_{1} + A_{2}B_{2} + ...+ A_{n}B_{n}$$

$2D Array.$

$$C_{i , j} = \sum\limits_{k \in [0 ... K) } (A_{i , k} . B_{k , j})$$

$$C_{i , j} = A_{i , k} . B_{k , j}$$

Consider

$$A \in \mathbb{R}^{m \times k}  ==   A \in \mathbb{R}^{rowA \times colA}$$

$$B \in \mathbb{R}^{k \times n}  ==   B \in \mathbb{R}^{rowB \times colB}$$

Code:

```c
#include <stdio.h>
#include <stdlib.h>

/*
Contant definition.

Here we use #define for the fan fact that it behave as macros whose value are subtituted at
run-time.
--------
Here we have the value size of rows and columns of matrix A[rowA][colA] and B[rowB][colB].

Note: Make sure if you change these values it has to obey the rule of matrix multiplication
of colA=rowB, A[N][K] . B[K][M]. Hence you will get C[rowA][colB] = C[N][M].
*/

#define rowA 4
#define colA 4
#define rowB 4
#define colB 4

/*
Declaration of dot_product() function.
-------------------------------------
This is going to return the Resulting Array of our dot product or matrix multiplication
*/

int** dot_product(int A[][colA], int B[][colB]);

/*
The main function
-----------------
*/

int main(){

    /*
    Note: If you will change rowA, colA, rowB and colB constant, make sure to change 
    also the 2-D arrays (Matricies) to match their size at their constant definition.
    */
    //Initialize and Declare the matrix A[rowA][colA]
    int A[][colA]={ {1,2,3,4},
                    {5,6,7,8}, 
                    {9,10,11,12},
                    {13,14,15,16}};

    //Initialize and Declare the matrix B[rowB][colB]
    int B[][colB]= {{1,2,3,4},
                    {1,2,3,4},
                    {1,2,3,4},
                    {1,2,3,4}};

    //Calling of dot_product() function and initialize it to a pointer variable int **mult
    int **mult=dot_product(A,B);

    //This is nested for-loop for printing the result matrix C[rowA][colB]
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            printf("%d\t", mult[i][j]);
        }
        printf("\n");
        
    }
    


    return 0;
}

/*
This is the dot_product() function.
            ------------
It takes in two 2D-arrays A[rowA][colA] and B[rowB][colB].

This function is said to return a pointer-to-point variable that points to the 
2-D array C[rowA][colB] memory block.
*/

int** dot_product (int A[][colA], int B[][colB]){

    //Pointer to pointer variable definition that will store C[rowA][colB]
    int ** mult=(int **)malloc(rowA*sizeof(int *));
    for (int i = 0; i < rowA; i++)
    {  
        mult[i] =(int *)malloc(colB*sizeof(int));
    }

    // Nested for-loop to perform Matrix Multiplication or Dot Product
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            mult[i][j]=0;
            for (int k = 0; k < colA; k++)
            {
                mult[i][j] += A[i][k]*B[k][j];
            }
        }  
    }
    return mult;
}
```