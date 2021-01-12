#include "helpers.h"

const int ERROR = -999;

// Kernel function is like the lambda inside a map()
// Each "tid = ..." is telling the tid-th thread to do something.
__global__ void map_carbon_edges_to_cycles(
  int* d_out, int* d_c_c, int d_c_c_size
) {
  // num_threads = blockDim.x * gridDim.x = 8 * 512 = 4096
  // Tells thread 0 to handle 0th, 4096th, (4096 * 2)th, ... entries
  // Tells thread 1 to handle 1st, (4096 + 1)th, (4096 * 2 + 1)th, ... entries
  // And so on.
  for (
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    tid < d_c_c_size;
    tid += blockDim.x * gridDim.x
  ) {
    // Look for 0th column, tid-th row in c-c
    int a = d_c_c[idx(0, tid, d_c_c_size)];
    // Look for 1st column, tid-th row in c-c
    int b = d_c_c[idx(1, tid, d_c_c_size)];
    int c = ERROR;
    int d = ERROR;
    int e = ERROR;
    int f = ERROR;

    for (int i = 0; i < d_c_c_size; i++) {
      if (d_c_c[idx(0, i, d_c_c_size)] == b) {
        int next = d_c_c[idx(1, i, d_c_c_size)];
        if (next != a && next != b)
          c = next;
        break;
      }
    }

    for (int i = 0; i < d_c_c_size; i++) {
      if (d_c_c[idx(0, i, d_c_c_size)] == c) {
        int next = d_c_c[idx(1, i, d_c_c_size)];
        if (next != a && next != b && next != c)
          d = next;
        break;
      }
    }

    for (int i = 0; i < d_c_c_size; i++) {
      if (d_c_c[idx(0, i, d_c_c_size)] == d) {
        int next = d_c_c[idx(1, i, d_c_c_size)];
        if (next != a && next != b && next != c && next != d)
          e = next;
        break;
      }
    }

    for (int i = 0; i < d_c_c_size; i++) {
      if (d_c_c[idx(0, i, d_c_c_size)] == e) {
        int next = d_c_c[idx(1, i, d_c_c_size)];
        if (next != a && next != b && next != c && next != d && next != e)
          f = next;
        break;
      }
    }

    bool is_a_f_joined = false;
    for (int i = 0; i < d_c_c_size; i++) {
      if (
        a == d_c_c[idx(0, i, d_c_c_size)] &&
        f == d_c_c[idx(1, i, d_c_c_size)]
      ) {
        is_a_f_joined = true;
        break;
      }
    }
    if (!is_a_f_joined)
      a = ERROR;

    // Injecting ith column, tid-th row in c-c, i = 0..5
    d_out[idx(0, tid, d_c_c_size)] = a;
    d_out[idx(1, tid, d_c_c_size)] = b;
    d_out[idx(2, tid, d_c_c_size)] = c;
    d_out[idx(3, tid, d_c_c_size)] = d;
    d_out[idx(4, tid, d_c_c_size)] = e;
    d_out[idx(5, tid, d_c_c_size)] = f;
  }
}

// Cannot call device function inside filter()
// since filter() is a host function
// So here define it one more time
int host_idx(int x, int y, int size) {
  return size * x + y;
}

void filter(
  int *filtered, int& filtered_size,
  int *in, int in_size
) {
  // Linearly scan each i-th row to see if we got valid cycle.
  filtered_size = 0;
  for (int i = 0; i < in_size; i++) {
    if (
      in[host_idx(0, i, in_size)] != ERROR &&
      in[host_idx(1, i, in_size)] != ERROR &&
      in[host_idx(2, i, in_size)] != ERROR &&
      in[host_idx(3, i, in_size)] != ERROR &&
      in[host_idx(4, i, in_size)] != ERROR &&
      in[host_idx(5, i, in_size)] != ERROR
    )
      filtered_size++;
  }

  // Inject valid cycle to filtered[] through counter magic :)
  for (int i = 0, j = 0; i < in_size; i++) {
    if (
      in[host_idx(0, i, in_size)] != ERROR &&
      in[host_idx(1, i, in_size)] != ERROR &&
      in[host_idx(2, i, in_size)] != ERROR &&
      in[host_idx(3, i, in_size)] != ERROR &&
      in[host_idx(4, i, in_size)] != ERROR &&
      in[host_idx(5, i, in_size)] != ERROR
    ) {
      filtered[host_idx(0, j, filtered_size)] = in[host_idx(0, i, in_size)];
      filtered[host_idx(1, j, filtered_size)] = in[host_idx(1, i, in_size)];
      filtered[host_idx(2, j, filtered_size)] = in[host_idx(2, i, in_size)];
      filtered[host_idx(3, j, filtered_size)] = in[host_idx(3, i, in_size)];
      filtered[host_idx(4, j, filtered_size)] = in[host_idx(4, i, in_size)];
      filtered[host_idx(5, j, filtered_size)] = in[host_idx(5, i, in_size)];
      j++;
    }
  }
}


void tnt_counting(
  int num_blocks_per_grid, int num_threads_per_block,
  int* c_c, int* c_n, int* c_h, int* n_o,
  int c_c_size, int c_n_size, int c_h_size, int n_o_size,
  int* &final_results, int &final_result_size
) {
  const int MAX_POINTER_SIZE = c_c_size * 100 * sizeof(int);

  int *d_c_c, *d_c_n, *d_c_h, *d_n_o;
  cudaMalloc(&d_c_c, 2 * c_c_size * sizeof(int));
  cudaMalloc(&d_c_n, 2 * c_n_size * sizeof(int));
  cudaMalloc(&d_c_h, 2 * c_h_size * sizeof(int));
  cudaMalloc(&d_n_o, 2 * n_o_size * sizeof(int));
  cudaMemcpy(d_c_c, c_c, 2 * c_c_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_n, c_n, 2 * c_n_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_h, c_h, 2 * c_h_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n_o, n_o, 2 * n_o_size * sizeof(int), cudaMemcpyHostToDevice);

  // BEGIN Kernel op
  int *d_result_1;
  cudaMalloc(&d_result_1, MAX_POINTER_SIZE);

  map_carbon_edges_to_cycles<<<num_blocks_per_grid, num_threads_per_block>>>(
    d_result_1, d_c_c, c_c_size
  );

  int *result_1 = (int *)malloc(MAX_POINTER_SIZE);
  int result_1_size = c_c_size;
  cudaMemcpy(result_1, d_result_1, MAX_POINTER_SIZE, cudaMemcpyDeviceToHost);
  // END Kernel op

  // Filter invalid cycles
  int *filtered_result_1 = (int *)malloc(MAX_POINTER_SIZE);
  int filtered_result_1_size;
  filter(
    filtered_result_1, filtered_result_1_size,
    result_1, result_1_size
  );

  // Filtered cycles
  final_results = (int *)malloc(MAX_POINTER_SIZE);
  final_result_size = filtered_result_1_size;
  for (int i = 0; i < final_result_size * 15; i++) {
    final_results[i] = filtered_result_1[i];
  }

  // Unfiltered cycles
  // final_results = (int *)malloc(MAX_POINTER_SIZE);
  // final_result_size = result_1_size;
  // for (int i = 0; i < final_result_size * 15; i++) {
  //   final_results[i] = result_1[i];
  // }

  cudaFree(d_c_c);
  cudaFree(d_c_n);
  cudaFree(d_c_h);
  cudaFree(d_n_o);
  cudaFree(d_result_1);

  free(result_1);
  free(filtered_result_1);
}
