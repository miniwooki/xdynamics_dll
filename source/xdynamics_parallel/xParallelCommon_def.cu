#include "xdynamics_parallel/xParallelCommon_decl.cuh"

//__device__ int calcGridHash(int3 gridPos, uint3 grid_size)
//{
//	gridPos.x = gridPos.x & (grid_size.x - 1);  // wrap grid, assumes size is power of 2
//	gridPos.y = gridPos.y & (grid_size.y - 1);
//	gridPos.z = gridPos.z & (grid_size.z - 1);
//	return __umul24(__umul24(gridPos.z, grid_size.y), grid_size.x) + __umul24(gridPos.y, grid_size.x) + gridPos.x;
//}
//
//// calculate position in uniform grid
//__device__ int3 calcGridPos(double3 p, double3 world_origin, double cell_size)
//{
//	int3 gridPos;
//	gridPos.x = floor((p.x - world_origin.x) / cell_size);
//	gridPos.y = floor((p.y - world_origin.y) / cell_size);
//	gridPos.z = floor((p.z - world_origin.z) / cell_size);
//	return gridPos;
//}

//void cudaMemoryAlloc(void** data, unsigned int size)
//{
//	checkCudaErrors(cudaMalloc(data, size));
//	checkCudaErrors(cudaMemset(*data, 0, size));
//}
//
//unsigned iDivUp(unsigned a, unsigned b)
//{
//	return (a % b != 0) ? (a / b + 1) : (a / b);
//}
//
//void computeGridSize(unsigned n, unsigned blockSize, unsigned &numBlocks, unsigned &numThreads)
//{
//	numThreads = min(blockSize, n);
//	numBlocks = iDivUp(n, numThreads);
//}
//
////void setSPHSymbolicParameter(device_sph_parameters *h_paras)
////{
////	checkCudaErrors(cudaMemcpyToSymbol(scte, h_paras, sizeof(device_sph_parameters)));
////}
//
//
//
//double __device__ maxfunc(double a, double b)
//{
//	return (b > a) ? b : a;
//}
//
//template <int BLOCKSIZE>
//void __global__ findMaxWithVector3(double3* inputvals, double* outputvals, int N)
//{
//	__shared__ volatile double data[BLOCKSIZE];
//	double maxval = sqrt(dot(inputvals[threadIdx.x]));
//	for (int i = blockDim.x + threadIdx.x; i < N; i += blockDim.x)
//	{
//		maxval = maxfunc(maxval, sqrt(dot(inputvals[i])));
//	}
//	data[threadIdx.x] = maxval;
//	__syncthreads();
//	if (threadIdx.x < 32) {
//		for (int i = 32 + threadIdx.x; i < BLOCKSIZE; i += 32){
//			data[threadIdx.x] = maxfunc(data[threadIdx.x], data[i]);
//		}
//		if (threadIdx.x < 16) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 16]);
//		if (threadIdx.x < 8) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 8]);
//		if (threadIdx.x < 4) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 4]);
//		if (threadIdx.x < 2) data[threadIdx.x] = maxfunc(data[threadIdx.x], data[threadIdx.x + 2]);
//		if (threadIdx.x == 0){
//			data[0] = maxfunc(data[0], data[1]);
//			outputvals[threadIdx.x] = data[0];
//		}
//	}
//}
//
//void cuMaxDouble3(double* indata, double* odata, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	computeGridSize(np, 512, numBlocks, numThreads);
//	findMaxWithVector3<512> << < numBlocks, numThreads >> >((double3 *)indata, odata, np);
//}
//
//template <typename T, unsigned int blockSize>
//__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
//{
//	/*extern*/ __shared__ T sdata[512];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//
//	T mySum = make_double3(0, 0, 0);;
//	//sdata[tid] = make_double3(0, 0, 0);
//
//	while (i < n)
//	{
//		//sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
//		mySum += g_idata[i];
//		if (i + blockSize < n)
//			mySum += g_idata[i + blockSize];
//		i += gridSize;
//	}
//	sdata[tid] = mySum;
//	__syncthreads();
//	if ((blockSize >= 512) && (tid < 256)) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
//	if ((blockSize >= 256) && (tid < 128)) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
//	if ((blockSize >= 128) && (tid < 64)) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
//	if ((blockSize >= 64) && (tid < 32)){ sdata[tid] = mySum = mySum + sdata[tid + 32]; } __syncthreads();
//	if ((blockSize >= 32) && (tid < 16)){ sdata[tid] = mySum = mySum + sdata[tid + 16]; } __syncthreads();
//
//	if ((blockSize >= 16) && (tid < 8))
//	{
//		sdata[tid] = mySum = mySum + sdata[tid + 8];
//	}
//
//	__syncthreads();
//
//	if ((blockSize >= 8) && (tid < 4))
//	{
//		sdata[tid] = mySum = mySum + sdata[tid + 4];
//	}
//
//	__syncthreads();
//
//	if ((blockSize >= 4) && (tid < 2))
//	{
//		sdata[tid] = mySum = mySum + sdata[tid + 2];
//	}
//
//	__syncthreads();
//
//	if ((blockSize >= 2) && (tid < 1))
//	{
//		sdata[tid] = mySum = mySum + sdata[tid + 1];
//	}
//
//	__syncthreads();
//
//	if (tid == 0) g_odata[blockIdx.x] = mySum;
//}
//
//double3 reductionD3(double3* in, unsigned int np)
//{
//	unsigned int numBlocks, numThreads;
//	double3 rt = make_double3(0.0, 0.0, 0.0);
//	computeGridSize(np, 512, numBlocks, numThreads);
//	double3* d_out;
//	double3* h_out = new double3[numBlocks];
//	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(double3) * numBlocks));
//	checkCudaErrors(cudaMemset(d_out, 0, sizeof(double3) * numBlocks));
//	//unsigned smemSize = sizeof(double3)*(512);
//	reduce6<double3, 512> << < numBlocks, numThreads/*, smemSize*/ >> >(in, d_out, np);
//	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
//	for (unsigned int i = 0; i < numBlocks; i++){
//		rt.x += h_out[i].x;
//		rt.y += h_out[i].y;
//		rt.z += h_out[i].z;
//	}
//	delete[] h_out;
//	checkCudaErrors(cudaFree(d_out));
//	return rt;
//}
//
