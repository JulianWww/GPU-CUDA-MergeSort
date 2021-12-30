#include <stdio.h>
#include <iostream>

#include <stdint.h>
#include <chrono>
#include <vector>

bool IsNotPowerOfTwo(const ulong x)
{
    return (x & (x - 1)) != 0;
}

static inline uint32_t getSteps(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y + (int)IsNotPowerOfTwo(x);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__
void GPUsortStep(int* x, int* y, int n, int sortSize)
{
    int i = (blockIdx.x*blockDim.x + threadIdx.x)*sortSize*2;
    int maxIter = i + sortSize*2;
    int maxAIter = i + sortSize;
    if (i < n){
        if (n < maxIter)
        {
            maxIter = n;
        }
        if (n <= maxAIter)
        {
            while (i < n)
            {
                y[i] = x[i];
                i ++;
            }
            return;
        }

        int iter = i;
        int iterA = i;
        int iterB = i + sortSize;
        if (iterB < n) 
        {
            while (iter < maxIter)
            {
                if (iterA >= maxAIter)
                {
                    y[iter] = x[iterB];
                    iterB ++;
                }
                else if (iterB >= maxIter)
                {
                    y[iter] = x[iterA];
                    iterA++;
                }
                else
                {
                    if (x[iterA] > x[iterB])
                    {
                        y[iter] = x[iterB];
                        iterB ++;
                    }
                    else
                    {
                        y[iter] = x[iterA];
                        iterA ++;
                    }
                }
                iter++;
            }
        }
    }
}

void CPUsortStep(int* x, int* y, int n, int sortSize, int i)
{
    int maxIter = i + sortSize*2;
    int maxAIter = i + sortSize;
    if (n < maxIter)
    {
        maxIter = n;
    }
    if (n <= maxAIter)
    {
        while (i < n)
        {
            y[i] = x[i];
            i ++;
        }
        return;
    }
    int iter = i;
    int iterA = i;
    int iterB = i + sortSize;
    if (iterB < n) 
    {
        while (iter < maxIter)
        {
            if (iterA >= maxAIter)
            {
                y[iter] = x[iterB];
                iterB ++;
            }
            else if (iterB >= maxIter)
            {
                y[iter] = x[iterA];
                iterA++;
            }
            else
            {
                if (x[iterA] > x[iterB])
                {
                    y[iter] = x[iterB];
                    iterB ++;
                }
                else
                {
                    y[iter] = x[iterA];
                    iterA ++;
                }
            }
            iter++;
        }
    }
}

void cudaSort(int * x,  int n, int steps, int& sorterSize)
{
    int *cuda_x, *cuda_y;
    gpuErrchk(cudaMalloc(&cuda_x, n*sizeof(int)));
    gpuErrchk(cudaMalloc(&cuda_y, n*sizeof(int)));

    gpuErrchk(cudaMemcpy(cuda_x, x, n*sizeof(int), cudaMemcpyHostToDevice));

    sorterSize = 1;
    for (size_t idx = 0; idx < steps; idx++)
    {
        const unsigned int threads = n / sorterSize;
        std::cout << idx << "\t" << threads << std::endl;
        if (threads < 256)
        {
            GPUsortStep<<<1, threads>>>(cuda_x, cuda_y, n, sorterSize);
        }
        else
        {
            GPUsortStep<<<(threads + 255)/256, 256>>>(cuda_x, cuda_y, n, sorterSize);
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sorterSize = sorterSize * 2;
        std::swap(cuda_x, cuda_y);
    }

    gpuErrchk(cudaMemcpy(x, cuda_x, n*sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(cuda_x));
    gpuErrchk(cudaFree(cuda_y));
}

int * cpuSort(int * x, int n, int steps, int size=10)
{
    int* y = (int*)malloc(sizeof(int)*n);
    for (size_t idx = 0; idx < steps; idx++)
    {
        size_t i = 0;
        while (i < n)
        {
            CPUsortStep(x, y, n, size, i);
            i += 2*size;
        }
        std::swap(x,y);
        size = size*2;
    }
    free(y);
    return x;
}

int * sort(int* x, int n, int cpuSteps = 10)
{
    int *out = (int*)malloc(n*sizeof(int));
    int sorterSize;
    auto steps = getSteps(n);
    cudaSort(x, n, steps-cpuSteps, sorterSize);
    return cpuSort(x, n, cpuSteps, sorterSize);
}

size_t test(int nums, int size)
{
  
  int N = 1<<size;
  std::cout << N * sizeof(int) << std::endl;
  int *x, *d_x, *d_y;
  x = (int*)malloc(N*sizeof(int));

  //cudaMalloc(&d_x, N*sizeof(int));
  //cudaMalloc(&d_y, N*sizeof(int)); 

  for (int i = 0; i < N; i++) {
    x[i] = std::rand();
  }
  std::cout << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  x = sort(x, N, nums);
  auto stop = std::chrono::high_resolution_clock::now();
  size_t timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

  for (int i = 0; i+1 < N; i++)
  {
    if (x[i+1] < x[i])
    {
        printf("error\n");
        return 1;
    }
    if(x[i] == 0){
        printf("NULL\n");
        return 1;
    }
  }

  free(x);

  std::cout << "sorting took: " << timeTaken << " ns" << std::endl;
  return timeTaken;
}

int main(void)
{
    std::vector<size_t> times = {};
    int N = 27;
    for (int idx = 0; idx < N; idx++)
    {
        times.push_back(test(idx, N));
    }
    for (auto const& val : times)
    {
        std::cout << val << ", ";
    }
    std::cout << std::endl;
}