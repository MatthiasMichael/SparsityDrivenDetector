#include "kernel_integralImage.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>

#include "cuda_error_check.h"


namespace sfs
{
	namespace cuda
	{
		__global__ void scan_horizontal(unsigned char * input, uint * output, uint imgWidth)
		{
			extern __shared__ uint temp[];

			//Pointer auf den Anfang der Zeile sezten, in der der Block arbeitet.
			output += blockIdx.y * imgWidth;

			//Inhalt des Input-Arrays für das der Block zuständig ist in das shared Array kopieren
			temp[2 * threadIdx.x] = input[blockIdx.y * imgWidth + 2 * threadIdx.x];
			temp[2 * threadIdx.x + 1] = input[blockIdx.y * imgWidth + 2 * threadIdx.x + 1];

			int offset = 1;

			//Reduktionsphase
			for (uint d = imgWidth >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;
					temp[index2] += temp[index1];
				}

				offset <<= 1;
			}

			//Letzten Wert in Temp wegschmeißen, da ein "Exclusive Integral Image" berechnet wird
			if (threadIdx.x == 0)
			{
				temp[imgWidth - 1] = 0;
			}

			//Down-Sweep Phase
			for (int d = 1; d < imgWidth; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;

					const int t = temp[index1];
					temp[index1] = temp[index2];
					temp[index2] += t;
				}
			}

			__syncthreads();

			//Ergebnisse aus dem Shared-Array in das Output-Array schreiben
			output[2 * threadIdx.x] = temp[2 * threadIdx.x];
			output[2 * threadIdx.x + 1] = temp[2 * threadIdx.x + 1];
		}


		__global__ void scan_horizontal_withoutPadding(unsigned char * input, uint * output, uint imgWidth, uint memoryWidth)
		{
			extern __shared__ uint temp[];

			//Inhalt des Input-Arrays für das der Block zuständig ist in das shared Array kopieren
			const int input1 = 2 * threadIdx.x < imgWidth ? input[blockIdx.y * imgWidth + 2 * threadIdx.x] : 0;
			const int input2 = 2 * threadIdx.x + 1 < imgWidth ? input[blockIdx.y * imgWidth + 2 * threadIdx.x + 1] : 0;

			temp[2 * threadIdx.x] = input1;
			temp[2 * threadIdx.x + 1] = input2;

			int offset = 1;

			//Reduktionsphase
			for (uint d = memoryWidth >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;
					temp[index2] += temp[index1];
				}

				offset <<= 1;
			}

			//Letzten Wert in Temp wegschmeißen, da ein "Exclusive Integral Image" berechnet wird
			if (threadIdx.x == 0)
			{
				temp[memoryWidth - 1] = 0;
			}

			//Down-Sweep Phase
			for (int d = 1; d < memoryWidth; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;

					const int t = temp[index1];
					temp[index1] = temp[index2];
					temp[index2] += t;
				}
			}

			__syncthreads();

			//Pointer auf den Anfang der Zeile sezten, in der der Block arbeitet.
			output += blockIdx.y * imgWidth;

			//Ergebnisse aus dem Shared-Array in das Output-Array schreiben
			if (2 * threadIdx.x < imgWidth)
			{
				output[2 * threadIdx.x] = temp[2 * threadIdx.x];
			}

			if (2 * threadIdx.x + 1 < imgWidth)
			{
				output[2 * threadIdx.x + 1] = temp[2 * threadIdx.x + 1];
			}
		}


		__global__ void scan_horizontal_withoutPadding_multiple(unsigned char * input, uint * output, uint imgWidth, uint imgHeight, uint memoryWidth)
		{
			extern __shared__ uint temp[];

			const uint offsetImage = blockIdx.z * imgWidth * imgHeight;

			//Inhalt des Input-Arrays für das der Block zuständig ist in das shared Array kopieren
			const int input1 = 2 * threadIdx.x < imgWidth ? input[offsetImage + blockIdx.y * imgWidth + 2 * threadIdx.x] : 0;
			const int input2 = 2 * threadIdx.x + 1 < imgWidth ? input[offsetImage + blockIdx.y * imgWidth + 2 * threadIdx.x + 1] : 0;

			temp[2 * threadIdx.x] = input1;
			temp[2 * threadIdx.x + 1] = input2;

			int offset = 1;

			//Reduktionsphase
			for (uint d = memoryWidth >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;
					temp[index2] += temp[index1];
				}

				offset <<= 1;
			}

			//Letzten Wert in Temp wegschmeißen, da ein "Exclusive Integral Image" berechnet wird
			if (threadIdx.x == 0)
			{
				temp[memoryWidth - 1] = 0;
			}

			//Down-Sweep Phase
			for (int d = 1; d < memoryWidth; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;

					const int t = temp[index1];
					temp[index1] = temp[index2];
					temp[index2] += t;
				}
			}

			__syncthreads();

			//Pointer auf den Anfang der Zeile sezten, in der der Block arbeitet.
			output += blockIdx.y * imgWidth;

			//Ergebnisse aus dem Shared-Array in das Output-Array schreiben
			if (2 * threadIdx.x < imgWidth)
			{
				output[offsetImage + 2 * threadIdx.x] = temp[2 * threadIdx.x];
			}

			if (2 * threadIdx.x + 1 < imgWidth)
			{
				output[offsetImage + 2 * threadIdx.x + 1] = temp[2 * threadIdx.x + 1];
			}
		}


		__global__ void scan_horizontal_withoutPadding_multiple_toSurface(unsigned char * input, cudaSurfaceObject_t output, uint imgWidth, uint imgHeight, uint memoryWidth)
		{
			extern __shared__ uint temp[];

			const uint offsetImage = blockIdx.z * imgWidth * imgHeight;

			//Inhalt des Input-Arrays für das der Block zuständig ist in das shared Array kopieren
			const int input1 = 2 * threadIdx.x < imgWidth ? input[offsetImage + blockIdx.y * imgWidth + 2 * threadIdx.x] : 0;
			const int input2 = 2 * threadIdx.x + 1 < imgWidth ? input[offsetImage + blockIdx.y * imgWidth + 2 * threadIdx.x + 1] : 0;

			temp[2 * threadIdx.x] = input1;
			temp[2 * threadIdx.x + 1] = input2;

			int offset = 1;

			//Reduktionsphase
			for (uint d = memoryWidth >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;
					temp[index2] += temp[index1];
				}

				offset <<= 1;
			}

			//Letzten Wert in Temp wegschmeißen, da ein "Exclusive Integral Image" berechnet wird
			if (threadIdx.x == 0)
			{
				temp[memoryWidth - 1] = 0;
			}

			//Down-Sweep Phase
			for (int d = 1; d < memoryWidth; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();

				if (threadIdx.x < d)
				{
					const int index1 = offset * (2 * threadIdx.x + 1) - 1;
					const int index2 = offset * (2 * threadIdx.x + 2) - 1;

					const int t = temp[index1];
					temp[index1] = temp[index2];
					temp[index2] += t;
				}
			}

			__syncthreads();

			//Ergebnisse aus dem Shared-Array in das Output-Array schreiben
			if (2 * threadIdx.x < imgWidth)
			{
				surf2Dwrite(temp[2 * threadIdx.x], output, (blockIdx.z * imgWidth + 2 * threadIdx.x) * sizeof(uint), blockIdx.y, cudaBoundaryModeZero);
			}

			if (2 * threadIdx.x + 1 < imgWidth)
			{
				surf2Dwrite(temp[2 * threadIdx.x + 1], output, (blockIdx.z * imgWidth + 2 * threadIdx.x + 1) * sizeof(uint), blockIdx.y, cudaBoundaryModeZero);
			}
		}


		void launchKernel_horizontalIntegralImg(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight)
		{
			if (imgWidth > 2 * 1024)
			{
				throw std::runtime_error("My current maximum block size is 1024. Can't process images with width larger than 2048");
			}

			dim3 blocksScanX(1, imgHeight);
			dim3 threadsScanX(imgWidth / 2, 1);

			scan_horizontal << <blocksScanX, threadsScanX, 2 * imgWidth * sizeof(int) >> > (dev_imgInput, dev_imgOutput, imgWidth);
			cudaDeviceSynchronize();

			cudaCheckError();
		}
			


		void launchKernel_horizontalIntegralImgWithoutPadding(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight)
		{
			if (imgWidth > 2 * 1024)
			{
				throw std::runtime_error("My current maximum block size is 1024. Can't process images with width larger than 2048");
			}

			const uint pow2ImgWidth = pow2roundup(imgWidth);

			dim3 blocksScanX(1, imgHeight);
			dim3 threadsScanX(pow2ImgWidth / 2, 1);

			scan_horizontal_withoutPadding << <blocksScanX, threadsScanX, 2 * pow2ImgWidth * sizeof(int) >> > (dev_imgInput, dev_imgOutput, imgWidth, pow2ImgWidth);

			cudaCheckError();
		}


		void launchKernel_horizontalIntegralImgWithoutPaddingMultiple(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight, uint numImages)
		{
			if (imgWidth > 2 * 1024)
			{
				throw std::runtime_error("My current maximum block size is 1024. Can't process images with width larger than 2048");
			}

			const uint pow2ImgWidth = pow2roundup(imgWidth);

			dim3 blocksScanX(1, imgHeight, numImages);
			dim3 threadsScanX(pow2ImgWidth / 2, 1, 1);

			scan_horizontal_withoutPadding_multiple << <blocksScanX, threadsScanX, 2 * pow2ImgWidth * sizeof(int) >> > (dev_imgInput, dev_imgOutput, imgWidth, imgHeight, pow2ImgWidth);
			cudaCheckError();
		}


		void launchKernel_horizontalIntegralImgWithoutPaddingMultipleToSurface(unsigned char * dev_imgInput, cudaSurfaceObject_t dev_output, uint imgWidth, uint imgHeight, uint numImages)
		{
			if (imgWidth > 2 * 1024)
			{
				throw std::runtime_error("My current maximum block size is 1024. Can't process images with width larger than 2048");
			}

			const uint pow2ImgWidth = pow2roundup(imgWidth);

			dim3 blocksScanX(1, imgHeight, numImages);
			dim3 threadsScanX(pow2ImgWidth / 2, 1, 1);

			scan_horizontal_withoutPadding_multiple_toSurface << <blocksScanX, threadsScanX, 2 * pow2ImgWidth * sizeof(int) >> > (dev_imgInput, dev_output, imgWidth, imgHeight, pow2ImgWidth);
			cudaCheckError();
		}
	}
}