/*
 * Do not modify this file
 */

#pragma once

/**
 * TYPE is the number of different types of bonds.
 * We only consider 4 types: c-c, c-n, c-h, n-o.
 */
#define TYPE 4

/**
 * EDGE_ATOM_SIZE is the number of dimension of each bond.
 * All original bonds have 2 atoms.
 */
#define EDGE_ATOM_SIZE 2

/**
 * number of vertices and edges in TNT
 */
#define NUM_TNT_VERTICES 15
#define NUM_TNT_EDGES 15

/**
 * Device function: Get flatten index of a 2-d array
 * x - row number
 * y - column number
 * size - size of each row
 */
inline __device__ int idx(int x, int y, int size) {
  return size * x + y;
}

void tnt_counting(
  int num_blocks_per_grid, int num_threads_per_block,
  int* c_c, int* c_n, int* c_h, int* n_o,
  int c_c_size, int c_n_size, int c_h_size, int n_o_size,
  int* &final_results, int &final_result_size
);
