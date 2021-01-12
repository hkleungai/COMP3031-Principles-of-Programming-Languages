// do NOT modify this file
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"

void print_matrix(void *payload)
{
    /* convert void* to Matrix* */
    Matrix *m = (Matrix*)payload;
    int i, j;
    for (i = 0; i < m->num_row; i++) 
    {
        for (j = 0; j < m->num_col; j++) 
        {
            printf("%d\t", m->data[i][j]);
        }
        printf("\n");
    }
}

/**
 * payload1: pointer to a matrix of size (m, n)
 * payload2: pointer to a matrix of size (1, n)
 * The function copy playload2 to the (m+1)th line of payload1
 * and return payload1 of size (m+1,n)
 */
void *append_row(void *payload1, void *payload2)
{
    /* convert void* to Matrix* */
    Matrix *m1 = (Matrix*)payload1;
    Matrix *m2 = (Matrix*)payload2;
    /* check the dimensions */
    if (m1->num_col != m2->num_col || m2->num_row != 1)
    {
        printf("dimensions not match!\n");
        exit(1);
    }

    /* copy elements */
    int i;
    for (i = 0; i < m1->num_col; i++)
    {
        m1->data[m1->num_row][i] = m2->data[0][i];
    }
    /* add number of rows */
    m1->num_row++;

    free(payload2);
    return payload1;
}

/**
 * payload1: pointer to a matrix of size (1, n)
 * payload2: pointer to a matrix of size (1, 1)
 * The function copy the element of playload2 to the position (1,n+1)
 * of payload1 and return payload1 with size (1,n+1)
 */
void *append_element(void *payload1, void *payload2)
{
    /* convert void* to Matrix* */
    Matrix *m1 = (Matrix*)payload1;
    Matrix *m2 = (Matrix*)payload2;
    /* check the dimensions */
    if (m1->num_row != 1 || m2->num_row != 1 || m2->num_col != 1)
    {
        printf("dimensions not match!\n");
        exit(1);
    }

    /* copy elements */
    m1->data[0][m1->num_col] = m2->data[0][0];
    /* add number of cols */
    m1->num_col++;

    free(payload2);
    return payload1;
}

/**
 * e: an integer
 * The function creates a matrix with size (1,1) with a single element 
 * and returns it
 */
void *element2matrix(int e)
{
    /* allocate a new matrix */
    Matrix *m = malloc(sizeof(Matrix*));
    m->data = (int**)malloc(sizeof(int*) * 16);
    int i;
    for (i = 0; i < 16; i++)
    {
        m->data[i] = (int*)malloc(sizeof(int) * 16);
    }

    /* set data */
    m->data[0][0] = e;
    /* set number of rows and cols */
    m->num_col = 1;
    m->num_row = 1;
    
    /* convert Matrix* to void* */
    return (void*)m;
}

/**
 * matrix add
 */
void *matrix_add(void *payload1, void *payload2)
{
    /* convert void* to Matrix* */
    Matrix *m1 = (Matrix*)payload1;
    Matrix *m2 = (Matrix*)payload2;
    /* check the dimensions */
    if (m1->num_row != m2->num_row || m1->num_col != m2->num_col)
    {
        printf("dimensions not match!\n");
        exit(1);
    }

    /* create result matrix */
    Matrix *m = malloc(sizeof(Matrix*));
    m->data = (int**)malloc(sizeof(int*) * 16);
    int i, j;
    for (i = 0; i < 16; i++)
    {
        m->data[i] = (int*)malloc(sizeof(int) * 16);
    }
    /* compute */
    for (i = 0; i < m1->num_row; i++)
    {
        for (j = 0; j < m1->num_col; j++)
        {
            m->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
    /* set number of rows and cols */
    m->num_row = m1->num_row;
    m->num_col = m1->num_col;

    free(payload1);
    free(payload2);
    /* convert Matrix* to void* */
    return (void*)m;
}

/**
 * matrix subtract
 */
void *matrix_sub(void *payload1, void *payload2)
{
    /* convert void* to Matrix* */
    Matrix *m1 = (Matrix*)payload1;
    Matrix *m2 = (Matrix*)payload2;
    /* check the dimensions */
    if (m1->num_row != m2->num_row || m1->num_col != m2->num_col)
    {
        printf("dimensions not match!\n");
        exit(1);
    }

    /* create result matrix */
    Matrix *m = malloc(sizeof(Matrix*));
    m->data = (int**)malloc(sizeof(int*) * 16);
    int i, j;
    for (i = 0; i < 16; i++)
    {
        m->data[i] = (int*)malloc(sizeof(int) * 16);
    }
    /* compute */
    for (i = 0; i < m1->num_row; i++)
    {
        for (j = 0; j < m1->num_col; j++)
        {
            m->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
    /* set number of rows and cols */
    m->num_row = m1->num_row;
    m->num_col = m1->num_col;

    free(payload1);
    free(payload2);
    /* convert Matrix* to void* */
    return (void*)m;
}

/**
 * matrix multiply
 */
void *matrix_mul(void *payload1, void *payload2)
{
    /* convert void* to Matrix* */
    Matrix *m1 = (Matrix*)payload1;
    Matrix *m2 = (Matrix*)payload2;
    /* check the dimensions */
    if (m1->num_col != m2->num_row)
    {
        printf("dimensions not match!\n");
        exit(1);
    }

    /* create result matrix */
    Matrix *m = malloc(sizeof(Matrix*));
    m->data = (int**)malloc(sizeof(int*) * 16);
    int i, j, k;;
    for (i = 0; i < 16; i++)
    {
        m->data[i] = (int*)malloc(sizeof(int) * 16);
    }
    /* compute */
    for (i = 0; i < m1->num_row; i++)
    {
        for (j = 0; j < m2->num_col; j++)
        {
            m->data[i][j] = 0;
            for (k = 0; k < m1->num_col; k++)
            {
                m->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    /* set number of rows and cols */
    m->num_row = m1->num_row;
    m->num_col = m2->num_col;

    free(payload1);
    free(payload2);
    /* convert Matrix* to void* */
    return (void*)m;
}
