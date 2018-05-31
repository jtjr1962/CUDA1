
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t searchWithCuda(int *found_index, int *found_thread_id, const int two_D_array_in[][5], int searchNumber, int x_size, int y_size);

//const int arraySize = 5;
const int numRows = 10;
const int numCols = 5;
const int s[numRows][numCols] =
{
	{ 10, 20, 30, 40, 50 },
	{ 11, 12, 13, 14, 15 },
	{ 21, 22, 23, 24, 25 },
	{ 31, 32, 33, 34, 35 },
	{ 41, 42, 43, 44, 45 },
	{ 51, 52, 53, 54, 55 },
	{ 61, 62, 63, 64, 65 },
	{ 71, 72, 73, 74, 75 },
	{ 81, 82, 83, 7, 85 },
	{ 91, 92, 93, 94, 95 },
};

__global__ void addKernel(int *c, const int *a, const int *b)
{   // this will run 1 time on each of the 5 GPU's selected
    int i = threadIdx.x;
	int stride = blockDim.x;
	printf("threadid.x=%d  threadid.y=%d threadid.z=%d  blockIdx.x= %d blockIdx.y= %d gridDim.x=%d gridDim.y=%d stride(blockdDim.x)=%d\n", i, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, stride);
    c[i] = a[i] + b[i];
}
 
// my attempt at strip search
// want number in, 2d array in; index out
__global__ void searchKernel(int *found_index, int *found_thread_id, int *fake_1D_array_in, int searchNumber, int pitch, int rows, int cols)
{   // this will run 1 time on each of the 5 GPU's selected
	int numElementsPerRow;
	int colIndex =0;

	printf("inside Kernel\n");
	printf("threadid.x=%d threadid.y=%d threadid.z=%d blockIdx.x=%d blockIdx.y=%d gridDim.x=%d gridDim.y=%d strid(blockdDim.x)=%d pitch=%d rows=%d cols=%d\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, blockDim.x, pitch, rows, cols);

	//	printf("searchNumber = %d  arraySize=%d\n", searchNumber, arraySize);
	numElementsPerRow = numCols;
	for (int x = threadIdx.x*numCols; colIndex < numElementsPerRow; x+=colIndex, colIndex++)
	{
		printf("threadid.x = %d fake_1D_array_in[x] == %d  x==%d\n", threadIdx.x, fake_1D_array_in[x],x);
		if (fake_1D_array_in[x] == searchNumber)
		{// only touch index if found; multiple hits not supported
			*found_index = x;
			*found_thread_id = threadIdx.x;
			printf("found it: found_index=%d   found_thread = %d\n", *found_index, *found_thread_id);
			break;
		}
		
	}

#if 0
	this is only to display elements, don't do processing like this with nested for loops!
	for (int r = 0; r < cols; ++r) {
		int* row = (int*)((char*)fake_1D_array_in + r * pitch);
		for (int c = 0; c < rows; ++c) {
			int element = row[c];
			printf("element= %d ", element);
			if (element == searchNumber)
			{// only touch index if found; multiple hits not supported
				*found_index = c*r;
				*found_thread_id = threadIdx.x;
				printf("found it: found_index=%d   found_thread = %d\n", found_index, found_thread_id);
				break;
			}
		}
		printf("end row\n");
	}

#endif
#if 0
	// want number in, 2d array in; index out
	__global__ void searchKernel(int *found_index, int *found_thread_id, int *fake_1D_array_in, int searchNumber, size_t pitch, size_t rows, size_t cols)
	{   // this will run 1 time on each of the 5 GPU's selected
		int i = threadIdx.x;
		int stride = blockDim.x;

		printf("threadid.x=%d  threadid.y=%d threadid.z=%d  blockIdx.x= %d blockIdx.y= %d gridDim.x==%d gridDim.y==%d stride(blockdDim.x)==%d\n",
			i, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, stride);

		printf("searchNumber = %d  arraySize=%d\n", searchNumber, arraySize);
		for (int x = threadIdx.x*arraySize; x < arraySize; x++)
		{
			printf("threadid.x = %d fake_1D_array_in[x] == %d\n", threadIdx.x, fake_1D_array_in[x]);
			if (fake_1D_array_in[x] == searchNumber)
			{// only touch index if found; multiple hits not supported
				*found_index = x;
				*found_thread_id = threadIdx.x;
				printf("found it: found_index=%d   found_thread = %d\n", found_index, found_thread_id);
				break;
			}
		}

	}
#endif

}
// want number in, stride in, 2d array in; index out
__global__ void searchStrideKernel(int *c, const int *a, const int *b)
{   // this will run 1 time on each of the 5 GPU's selected
	int i = threadIdx.x;
	int stride = blockDim.x;
	printf("threadid=%d stride==%d\n", i, stride);
	c[i] = a[i] + b[i];
}
/**********************************************************************
  Main ()

***********************************************************************/
int main()
{
	int main_found_index = 32767;
	int main_found_thread = 32767;
	const int number_to_search_for = 7;
	cudaError_t cudaStatus;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };

//		int c[arraySize] = { 0 };
#if 0
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

#endif
	
	//	cudaError_t searchWithCuda(int *found_index, int *found_thread_id, const int two_D_array_in[][5], int searchNumber, size_t x_size, size_t y_size)
	cudaStatus = searchWithCuda(&main_found_index, &main_found_thread,                              s, number_to_search_for,            10, 5);
//	cudaError_t searchWithCuda(int *found_index, int *found_thread_id, const int two_D_array_in[][5],      int searchNumber, size_t x_size, size_t y_size);

	
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


// Helper function for using CUDA to search array in parallel.

cudaError_t searchWithCuda(int *found_index, int *found_thread_id, const int two_D_array_in[][5], int searchNumber, int x_size, int y_size)
{

	int rows, cols;
	size_t pitch;

	int *dev_s = 0;

	int *dev_d_managed = 0;

	int *found_index_managed = 0;
	int *found_index_thread = 0;

	cudaError_t cudaStatus;

	rows = y_size;
	cols = x_size;
	printf("echo args\n int *found_index= %d, int *found_thread_id= %d, skipping in array, int searchNumber= %d, size_t x_size= %d, size_t y_size= %d\n",
		                  *found_index,               *found_thread_id,                            searchNumber,            x_size,           y_size);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	// set pointer to dev_s and get pitch at that memory based on rows and columns
	cudaStatus = cudaMallocPitch((void**)&dev_s, &pitch, sizeof(int)*cols, rows);

	cudaMemcpy2D(dev_s, pitch, s, sizeof(int)*cols, sizeof(int)*cols, rows, cudaMemcpyHostToDevice);

#if 0
	int * dev_s_ptr = 0;
	cudaMemcpy2D(dev_s_ptr, pitch, dev_s, sizeof(int)*cols, sizeof(int)*cols, rows, cudaMemcpyDeviceToHost);
	//	fixit echo back array

	for (int index = 0; index < rows*cols; index++)
	{
		printf("%d", *dev_s_ptr);
		dev_s_ptr++;
		if (index%arraySize == 0)
			printf("\n");
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

#endif // 0

	// Allocate Unified Memory - accessible from either GPU or CPU.
	cudaStatus = cudaMallocManaged((void**)&dev_d_managed, x_size * y_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed!");
		goto Error;
	}

	// Allocate Unified Memory - accessible from either GPU or CPU.
	cudaStatus = cudaMallocManaged((void**)&found_index_managed, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged found_index_managed failed!");
		goto Error;
	}
	*found_index_managed = 32726;

	// Allocate Unified Memory - accessible from either GPU or CPU.
	cudaStatus = cudaMallocManaged((void**)&found_index_thread, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged found_index_managed failed!");
		goto Error;
	}
	*found_index_thread = 17777;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_s, x_size * y_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_s, two_D_array_in, x_size * y_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//kernelopname  <<<gridDim.x,  stride >>> ( args)
	printf("searchKernel call rows=%d\n", rows);
	//             1st arg
	//             caause blockIdx.x
	//             to go 0 to arg
//	__global__ void searchKernel(      int *found_index, int *found_thread_id, int *fake_1D_array_in, int searchNumber, size_t pitch, size_t rows, size_t cols)
//	searchKernel << <1, x_size >> >(found_index_managed, found_index_thread, dev_s, searchNumber, pitch, rows, cols);
	searchKernel << <1, x_size >> >(found_index_managed, found_index_thread,                   dev_s,     searchNumber,        pitch,         rows,        cols);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching searchKernel\n", cudaStatus);
		goto Error;
	}

	printf ("found_index_managed=%d found_index_thread=%d\n", *found_index_managed, *found_index_thread);
#if 0
	only returning index, not data
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#endif

Error:
	cudaFree(dev_s);
	cudaFree(dev_d_managed);

	return cudaStatus;
}

#if 0
not used
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	int *dev_d_managed = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate Unified Memory - accessible from either GPU or CPU.
	cudaStatus = cudaMallocManaged((void**)&dev_d_managed, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread *for each* element (5 threads in original case (arraySize==5)
	//         5 = arraySize
	//	printf("addKernel<<<1, size>>>(dev_c, dev_a, dev_b);\n");
	//	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	//kernelopname  <<<gridDim.x,  stride >>> ( args)
	printf("addKernel <<<4, size >>>(dev_c, dev_a, dev_b);\n");
	addKernel << <4, size >> >(dev_c, dev_a, dev_b);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
#endif