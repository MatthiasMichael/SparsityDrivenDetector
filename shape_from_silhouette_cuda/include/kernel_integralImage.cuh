#pragma once

#include <vector_functions.h>

#include "cuda_math_utils.h"

namespace sfs
{
	namespace cuda
	{
		void launchKernel_horizontalIntegralImg(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight);
		void launchKernel_horizontalIntegralImgWithoutPadding(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight);
		void launchKernel_horizontalIntegralImgWithoutPaddingMultiple(unsigned char * dev_imgInput, uint * dev_imgOutput, uint imgWidth, uint imgHeight, uint numImages);

		void launchKernel_horizontalIntegralImgWithoutPaddingMultipleToSurface(unsigned char * dev_imgInput, cudaSurfaceObject_t dev_output, uint imgWidth, uint imgHeight, uint numImages);


		inline uint pow2roundup(uint x)
		{
			--x;

			x |= x >> 1;
			x |= x >> 2;
			x |= x >> 4;
			x |= x >> 8;
			x |= x >> 16;

			return x + 1;
		}
	}
}