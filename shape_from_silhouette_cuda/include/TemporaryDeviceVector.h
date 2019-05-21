#pragma once

#ifndef _CUDACC_
#include <cassert>
#endif

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

namespace sfs
{
	namespace cuda
	{
		/**
		 * Intended for use in Cuda-Kernels to pass raw pointer around without having to use an additional parameter for size.
		 * Only wraps around raw pointers.
		 */
		template <typename T>
		class TemporaryDeviceVector
		{
		public:
			__host__ __device__ TemporaryDeviceVector() : m_begin(), m_end(), m_size(0)
			{
				// empty
			}

			__host__ __device__ TemporaryDeviceVector(const T * begin, const T * end) :
				m_begin(begin), m_end(end), m_size(end - begin)
			{
				// empty
			}

			__host__ __device__ TemporaryDeviceVector(const T * begin, const size_t size) :
				m_begin(begin), m_end(begin + size), m_size(size)
			{
				// empty
			}

			__host__ __device__ T & operator[](size_t idx)
			{
#ifndef _CUDACC_
				assert(idx < m_size);
#endif
				return *(m_begin + idx);
			}

			__host__ __device__ const T & operator[](size_t idx) const
			{
#ifndef _CUDACC_
				assert(idx < m_size);
#endif
				return *(m_begin + idx);
			}

			__host__ __device__ size_t size() const { return m_size; }

		private:
			const T * m_begin;
			const T * m_end;

			size_t m_size;
		};
	}
}
