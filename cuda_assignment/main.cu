/*
 * Do not modify this file
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <string>
#include <chrono>

#include "helpers.h"

/**
 * Read file, save edges to array (x_x) and
 * record the size of each type of edge array (x_x_count).
 */
int read_file(
  std::string filename, int* &c_c, int* &c_n, int* &c_h, int* &n_o,
  int &c_c_size, int &c_n_size, int &c_h_size, int &n_o_size
) {
  const char edge_endpoints[TYPE][EDGE_ATOM_SIZE + 1] = {
      {'c', 'c'}, {'c', 'n'}, {'c', 'h'}, {'n', 'o'}
  };
  std::ifstream inputf(filename, std::ifstream::in);

  for (int type = 0; type < TYPE; type++) {
    char s_atom, t_atom;
    int size;
    inputf >> s_atom >> t_atom >> size;
    assert(
      s_atom == edge_endpoints[type][0] &&
      t_atom == edge_endpoints[type][1] &&
      "Input file format error!"
    );

    switch (type) {
      case 0: /* c-c */
        /**
          * For c-c edges, the source and target atom is the same,
          * so we also store the reversed edges.
          * e.g. if there is an edge (1,2), we also store the edge (2,1).
          */
        c_c_size = size * 2;
        c_c = (int *)malloc(sizeof(int) * c_c_size * 2);
        for (int i = 0; i < c_c_size / 2; i++) {
            inputf >> c_c[i] >> c_c[i + c_c_size];
            c_c[c_c_size / 2 + i] = c_c[i + c_c_size];
            c_c[c_c_size / 2 + c_c_size + i] = c_c[i];
        }
        break;
      case 1: /* c-n */
        c_n_size = size;
        c_n = (int *)malloc(sizeof(int) * c_n_size * 2);
        for (int i = 0; i < c_n_size; i++) {
            inputf >> c_n[i] >> c_n[i + c_n_size];
        }
        break;
      case 2: /* c-h */
        c_h_size = size;
        c_h = (int *)malloc(sizeof(int) * c_h_size * 2);
        for (int i = 0; i < c_h_size; i++) {
            inputf >> c_h[i] >> c_h[i + c_h_size];
        }
        break;
      case 3: /* n-o */
        n_o_size = size;
        n_o = (int *)malloc(sizeof(int) * n_o_size * 2);
        for (int i = 0; i < n_o_size; i++) {
            inputf >> n_o[i] >> n_o[i + n_o_size];
        }
        break;
    }
  }
  inputf.close();
  return 0;
}

int main(int argc, char **argv) {
  assert(argc == 4 && "Input format error!");
  std::string filename = argv[1];
  int num_blocks_per_grid = atoi(argv[2]);
  int num_threads_per_block = atoi(argv[3]);

  /**
    * Define 4 types of bounds and their size.
    * Each x_x is a 2-d array, but we save it in a flatten manner.
    * We can find the element at the i-th row and j-th column by
    * accessing x_x[i * size + j], where size is the row length
    */
  int *c_c, *c_n, *c_h, *n_o;
  int c_c_size, c_n_size, c_h_size, n_o_size;

  assert(read_file(
    filename, c_c, c_n, c_h, n_o,
    c_c_size, c_n_size, c_h_size, n_o_size
  ) == 0);

#ifdef DEBUG
  std::cout << "c-c:" << std::endl;
  for (int i = 0; i < c_c_size; i++) {
    std::cout << c_c[i] << "-" << c_c[i + c_c_size] << ' ';
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl << "c-n:" << std::endl;
  for (int i = 0; i < c_n_size; i++) {
    std::cout << c_n[i] << "-" << c_n[i + c_n_size] << ' ';
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl <<  "c-h:" << std::endl;
  for (int i = 0; i < c_h_size; i++) {
    std::cout << c_h[i] << "-" << c_h[i + c_h_size] << ' ';
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl << "n-o:" << std::endl;
  for (int i = 0; i < n_o_size; i++) {
    std::cout << n_o[i] << "-" << n_o[i + n_o_size] << ' ';
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
#endif

  cudaDeviceReset();
  auto t_start = std::chrono::high_resolution_clock::now();

  cudaEvent_t cuda_start, cuda_end;
  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_end);
  float kernel_time;

  cudaEventRecord(cuda_start);

  int *final_results = nullptr, final_result_size = 0;

  tnt_counting(
    num_blocks_per_grid, num_threads_per_block,
    c_c, c_n, c_h, n_o,
    c_c_size, c_n_size, c_h_size, n_o_size,
    final_results, final_result_size
  );

  cudaEventRecord(cuda_end);

  cudaEventSynchronize(cuda_start);
  cudaEventSynchronize(cuda_end);
  cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);

  cudaDeviceSynchronize();

  auto t_end = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < final_result_size; i++) {
    for (int j = 0; j < 15; j++) {
      std::cout << final_results[j * final_result_size + i] << " ";
    }
    std::cout << std::endl;
  }

  /* print out the results and the time consumption */
  fprintf(stderr, "Number of Results: %d\n", final_result_size);

  fprintf(
    stderr, "Elapsed Time: %.9lf s\n",
    std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / pow(10, 9)
  );
  fprintf(
    stderr, "Driver Time: %.9lf s\n",
    kernel_time / pow(10, 3)
  );

  std::cout << "Mine's here" << std::endl;

  free(final_results);
  free(c_c);
  free(c_n);
  free(c_h);
  free(n_o);

  return 0;
}
