// do NOT modify this file
#ifndef __HEADER_GLOBAL__
#define __HEADER_GLOBAL__
#include <stdio.h>

typedef struct Matrix{
    int **data;
    int num_row;
    int num_col;
}Matrix;

/* these are functions you may use in matcal.y */
void print_matrix(void *payload);
void *append_row(void *payload1, void *payload2);
void *append_element(void *payload1, void *payload2);
void *element2matrix(int e);
void *matrix_add(void *payload1, void *payload2);
void *matrix_sub(void *payload1, void *payload2);
void *matrix_mul(void *payload1, void *payload2);

#endif