#pragma once
#include <cuda.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

//Error Function
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

//SM GROUP INFO
std::vector<std::vector<int>> sm_groups = {
        {0,4,8,12,16},
        {1,5,9,13,17},
        {2,6,10,14,18},
        {3,7,11,15,19}
};


//Policy structure
typedef struct
{
  char *sm_id;
  int *block_id;
  int *block_miss;
  int block_num;
  int base_block_id;
  int max_miss_num;
  dim3 grid;
} Policy;



class App
{
private:
public:
  std::vector<int> task_smid;
  static const int SM_COUNT = 20;
  bool priority;
  char *sm_id;
  char *all_sm;
  int thread_num;
  int sm_group_id;
  App() {}
 
  App(int id) : sm_group_id(id)
  {
    task_smid = sm_groups[id];
    thread_num = 0;
    cudaMalloc((void **)&sm_id, sizeof(char) * SM_COUNT);
    cudaMalloc((void **)&all_sm, sizeof(char) * SM_COUNT);
    cudaMemset(sm_id, 0, sizeof(char) * SM_COUNT);
    cudaMemset(all_sm, 1, sizeof(char) * SM_COUNT);
    for (const auto i : task_smid)
      cudaMemset(sm_id + i, 1, sizeof(char));
  }
  
  ~App()
  {
    cudaFree(sm_id);
    cudaFree(all_sm);
  }
};

cudaStream_t getStream()
{
  cudaStream_t t;
  cudaStreamCreate(&t);
  return t;
}


__device__ char runable_retreat(int *bid, int *miss_num, Policy con)
{
    
// get SM_ID of the register being used
  unsigned int ret;
  asm("mov.u32 %0, %smid;"
      : "=r"(ret));
    
//if it is the same as the assigned SM_ID continue, else return
  if (!con.sm_id[ret])
    if (threadIdx.x == 0)
      miss_num[0] = atomicAdd((int *)con.block_miss, 1);

  __syncthreads();

  if (!con.sm_id[ret])
    if (miss_num[0] < con.max_miss_num)
      return 0;
  __syncthreads();
  if (threadIdx.x == 0)
    bid[0] = atomicAdd((int *)con.block_id, 1);
  __syncthreads();
  if (bid[0] >= con.grid.x + con.base_block_id)
    return 0;
  __syncthreads();
  return 1;
}

__device__ char runable_retreat(int *bidx, int *bidy, int *miss_num, Policy con)
{
// get SM_ID of the register being used
  unsigned int ret;
  asm("mov.u32 %0, %smid;"
      : "=r"(ret));
    
  if (!con.sm_id[ret])
    if (threadIdx.x == 0 && threadIdx.y == 0)
      miss_num[0] = atomicAdd((int *)con.block_miss, 1);
  __syncthreads();
  if (!con.sm_id[ret])
    if (miss_num[0] < con.max_miss_num)
      return 0;
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    int t = atomicAdd((int *)con.block_id, 1);
    bidy[0] = t / con.grid.x;
    bidx[0] = t % con.grid.x;

  }
  __syncthreads();
  if (bidx[0] >= con.grid.x || bidy[0] >= con.grid.y)
    return 0;
  return 1;
}

//returns the information as a Policy structure
Policy dispatcher(const App &app, const dim3 &grid, const dim3 &block,
                  int base_block_id = 0)
{
  Policy t;
  int block_num = grid.x * grid.y * grid.z;
  cudaMalloc((void **)&t.block_id, sizeof(int));
  cudaMemcpy(t.block_id, &base_block_id, sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&t.block_miss, sizeof(int));
  int block_miss = 0;
  cudaMemcpy(t.block_miss, &block_miss, sizeof(int), cudaMemcpyHostToDevice);
  t.sm_id = app.sm_id;
  t.base_block_id = base_block_id;
  t.grid = grid;
  t.block_num =
      ceil((float)block_num * App::SM_COUNT / (float)app.task_smid.size());
  t.max_miss_num = t.block_num - block_num;
  return t;
}

void task_destroy(Policy &t)
{
  cudaFree(t.block_id);
  cudaFree(t.block_miss);
}

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
