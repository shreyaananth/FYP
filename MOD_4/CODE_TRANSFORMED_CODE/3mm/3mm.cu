/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "../../runtime.cuh"
#include "../../common/polybenchUtilFuncts.cuh"
#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NI 64
#define NJ 64
#define NK 64
#define NL 64
#define NM 64

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D)
{
  int i, j;
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NK; j++)
    {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NJ; i++)
  {
    for (j = 0; j < NM; j++)
    {
      C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NM; i++)
  {
    for (j = 0; j < NL; j++)
    {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NL; j++)
    {
      if (percentDiff(G[i * NL + j], G_outputFromGpu[i * NL + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD)
      {
        fail++;
      }
    }
  }

  // print results
  //printf(
  //    "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
  //    "%d\n",
   //   PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E,
                            Policy con)
{
  __shared__ int bidx[1];
  __shared__ int bidy[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bidx, bidy, miss_num, con))
    return;
  int j = bidx[0] * blockDim.x + threadIdx.x;
  int i = bidy[0] * blockDim.y + threadIdx.y;

  if ((i < NI) && (j < NJ))
  {
    int k;
    for (k = 0; k < NK; k++)
    {
      E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
    }
  }
}

__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F,
                            Policy con)
{
  __shared__ int bidx[1];
  __shared__ int bidy[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bidx, bidy, miss_num, con))
    return;
  int j = bidx[0] * blockDim.x + threadIdx.x;
  int i = bidy[0] * blockDim.y + threadIdx.y;

  if ((i < NJ) && (j < NL))
  {
    int k;
    for (k = 0; k < NM; k++)
    {
      F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
    }
  }
}

__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G,
                            Policy con)
{
  __shared__ int bidx[1];
  __shared__ int bidy[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bidx, bidy, miss_num, con))
    return;

  int j = bidx[0] * blockDim.x + threadIdx.x;
  int i = bidy[0] * blockDim.y + threadIdx.y;

  if ((i < NI) && (j < NL))
  {
    int k;
    for (k = 0; k < NJ; k++)
    {
      G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
    }
  }
}

void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
  int i, j, k;

  /* E := A*B */
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NJ; j++)
    {
      E[i * NJ + j] = 0;
      for (k = 0; k < NK; ++k)
      {
        E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  /* F := C*D */
  for (i = 0; i < NJ; i++)
  {
    for (j = 0; j < NL; j++)
    {
      F[i * NL + j] = 0;
      for (k = 0; k < NM; ++k)
      {
        F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
      }
    }
  }

  /* G := E*F */
  for (i = 0; i < NI; i++)
  {
    for (j = 0; j < NL; j++)
    {
      G[i * NL + j] = 0;
      for (k = 0; k < NJ; ++k)
      {
        G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
      }
    }
  }
}

void mm3Cuda(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G,
             DATA_TYPE *G_outputFromGpu, const int &app_id)
{
  DATA_TYPE *A_gpu;
  DATA_TYPE *B_gpu;
  DATA_TYPE *C_gpu;
  DATA_TYPE *D_gpu;
  DATA_TYPE *E_gpu;
  DATA_TYPE *F_gpu;
  DATA_TYPE *G_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
  cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
  cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
  cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
  cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
  cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
  cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
  cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
  cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
  cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
  cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X))),
             (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));
  dim3 grid2((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))),
             (size_t)(ceil((float)NJ / ((float)DIM_THREAD_BLOCK_Y))));
  dim3 grid3((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))),
             (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));
 
  //----------------------------------------------------
 
  App app(app_id);
  
  Policy task1 = dispatcher(app, grid1, block);
  Policy task2 = dispatcher(app, grid2, block);
  Policy task3 = dispatcher(app, grid3, block);
 
 double t_start, t_end;
  t_start = rtclock();
  
   //kernel_splitting_send(app_id);
  mm3_kernel1<<<task1.block_num, block>>>(A_gpu, B_gpu, E_gpu, task1);
  mm3_kernel2<<<task2.block_num, block>>>(C_gpu, D_gpu, F_gpu, task2);
  mm3_kernel3<<<task3.block_num, block>>>(E_gpu, F_gpu, G_gpu, task3);
  
    //kernel_splitting_receive(app_id);

  t_end = rtclock();
  
   fprintf(stdout, "%0.6lf\n", 
          (t_end - t_start) * 1000);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  
  task_destroy(task1);
  task_destroy(task2);
  task_destroy(task3);
  
  
  cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL,
             cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  cudaFree(D_gpu);
  cudaFree(E_gpu);
  cudaFree(F_gpu);
  cudaFree(G_gpu);
}

int main(int argc, char **argv)
{
  int app_id = atoi(argv[1]);
  double t_start, t_end;
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *F;
  DATA_TYPE *G;
  DATA_TYPE *G_outputFromGpu;
  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NJ * NM * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NM * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  F = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  G = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
  G_outputFromGpu = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));

  init_array(A, B, C, D);


  mm3Cuda(A, B, C, D, E, F, G, G_outputFromGpu, app_id);

  t_start = rtclock();

  mm3_cpu(A, B, C, D, E, F, G);

  t_end = rtclock();

  //fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(G, G_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(F);
  free(G);
  free(G_outputFromGpu);

  return 0;
}
