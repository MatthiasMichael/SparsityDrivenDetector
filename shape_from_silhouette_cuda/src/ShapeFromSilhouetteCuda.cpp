#include "ShapeFromSilhouetteCuda.h"

#include <algorithm>
#include <numeric>
#include <fstream>

#include <cuda_runtime_api.h>

#include "cuda_error_check.h"

#include "ApplicationTimer.h"

#include "kernel_integralImage.cuh"


namespace sfs
{
	namespace cuda
	{
		struct ShapeFromSilhouette::StringConstants
		{
			inline static const auto MissingSpace = "Create space before accessing values!";

			inline static const auto TP_COPY = "ShapeFromSilhouette_Cuda::processInput::Copy";
			inline static const auto TP_INTEGRAL = "ShapeFromSilhouette_Cuda::processInput::IntegralImage";
			inline static const auto TP_UPDATE = "ShapeFromSilhouette_Cuda::processInput::UpdateSpace";
			inline static const auto TP_FILL = "ShapeFromSilhouette_Cuda::processInput::FillSpace";
			inline static const auto TP_TOTAL = "ShapeFromSilhouette_Cuda::processInput";
		};


		ShapeFromSilhouette::Parameters::Parameters() :
			minSegmentation(0.05f),
			maxClusterDistance(3)
		{
			// empty
		}


		ShapeFromSilhouette::ShapeFromSilhouette() :
			m_space(nullptr),
			m_imageSize{ 0, 0 },
			m_numImages(0),
			m_dev_imagesSegmentation(nullptr),
			m_dev_integralImages(nullptr),
			m_integralImageSurface(0),
			m_integralImagesArray(nullptr),
			m_integralImageSurfaceChannelDesc(
			cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned))
		{
			memset(&m_integralImageSurfaceResourceDesc, 0, sizeof(m_integralImageSurfaceResourceDesc));
		}


		ShapeFromSilhouette::~ShapeFromSilhouette()
		{
			freeDeviceMemory();

			m_space.reset(nullptr); // Otherwise it will be attempted to delete it after the device reset

			cudaDeviceSynchronize();
			//cudaDeviceReset();
		}


		ShapeFromSilhouette::ShapeFromSilhouette(ShapeFromSilhouette && o) noexcept :
			m_space(std::move(o.m_space)),
			m_imageSize(std::move(o.m_imageSize)),
			m_numImages(std::move(o.m_numImages)),
			m_dev_imagesSegmentation(std::move(o.m_dev_imagesSegmentation)),
			m_dev_integralImages(std::move(o.m_dev_integralImages)),
			m_integralImageSurface(std::move(o.m_integralImageSurface)),
			m_integralImagesArray(std::move(o.m_integralImagesArray)),
			m_integralImageSurfaceChannelDesc(std::move(o.m_integralImageSurfaceChannelDesc)),
			m_integralImageSurfaceResourceDesc(std::move(o.m_integralImageSurfaceResourceDesc))
		{
			o.m_dev_imagesSegmentation = nullptr;
			o.m_dev_integralImages = nullptr;

			o.m_integralImageSurface = cudaSurfaceObject_t{ 0 };
			o.m_integralImagesArray = nullptr;

			memset(&o.m_integralImageSurfaceResourceDesc, 0, sizeof(o.m_integralImageSurfaceResourceDesc));
		}


		void ShapeFromSilhouette::loadSpace(const std::string & filename)
		{
			std::fstream filestream(filename, std::fstream::in | std::fstream::binary);

			m_space.reset(new Space());
			readBinary(filestream, *m_space);
		}


		void ShapeFromSilhouette::processInput(const std::vector<cv::Mat> & imagesSegmentation)
		{
			cudaDeviceSynchronize();

			AT_START(StringConstants::TP_TOTAL);

			AT_START(StringConstants::TP_COPY);
			copyInput(imagesSegmentation);
			AT_STOP(StringConstants::TP_COPY);

			AT_START(StringConstants::TP_INTEGRAL);
			launchKernel_horizontalIntegralImgWithoutPaddingMultipleToSurface(
				m_dev_imagesSegmentation, m_integralImageSurface, m_imageSize.x, m_imageSize.y, 
				static_cast<uint>(m_numImages));
			cudaDeviceSynchronize();
			AT_STOP(StringConstants::TP_INTEGRAL);

			AT_START(StringConstants::TP_UPDATE);
			m_space->updateFromSurfaceIntegralImage(m_integralImageSurface, make_uint2(m_imageSize.x, m_imageSize.y),
				static_cast<uint>(imagesSegmentation.size()), m_parameters.minSegmentation);

			//m_space->update(m_dev_imagesSegmentation, make_uint2(m_imageSize.x, m_imageSize.y), imagesSegmentation.size(), m_parameters.minSegmentation);
			AT_STOP(StringConstants::TP_UPDATE);

			AT_START(StringConstants::TP_FILL);
			m_clusterObjects = m_space->sequentialFill(m_parameters.maxClusterDistance);
			AT_STOP(StringConstants::TP_FILL);

			AT_STOP(StringConstants::TP_TOTAL);
		}


		const CameraSet & ShapeFromSilhouette::getCameraModels() const
		{
			if (!hasSpace())
			{
				throw std::runtime_error(StringConstants::MissingSpace);
			}

			return m_space->getCameraModels();
		}


		Roi3DF ShapeFromSilhouette::getArea() const
		{
			if (!hasSpace())
			{
				throw std::runtime_error(StringConstants::MissingSpace);
			}

			return m_space->getArea();
		}


		const Space & ShapeFromSilhouette::getSpace() const
		{
			if (!hasSpace())
			{
				throw std::runtime_error(StringConstants::MissingSpace);
			}

			return *m_space;
		}


		bool ShapeFromSilhouette::needDeviceInitialization(const std::vector<cv::Mat> & imagesSegmentation) const
		{
			assert(imagesSegmentation.size() != 0);

			return m_numImages != imagesSegmentation.size() ||
				m_imageSize.x != imagesSegmentation[0].size().width ||
				m_imageSize.y != imagesSegmentation[0].size().height;
		}


		void ShapeFromSilhouette::initDeviceMemory(const std::vector<cv::Mat> & imagesSegmentation)
		{
			m_numImages = imagesSegmentation.size();
			m_imageSize.x = imagesSegmentation[0].size().width;
			m_imageSize.y = imagesSegmentation[0].size().height;

			const size_t completeSize = m_numImages * m_imageSize.x * m_imageSize.y;

			cudaSafeCall(cudaMalloc((void**)&m_dev_imagesSegmentation, completeSize * sizeof(unsigned char)));
			cudaSafeCall(cudaMalloc((void**)&m_dev_integralImages, completeSize * sizeof(uint)));

			cudaSafeCall(cudaMallocArray(&m_integralImagesArray, &m_integralImageSurfaceChannelDesc, m_numImages *
				m_imageSize.x, m_imageSize.y, cudaArraySurfaceLoadStore));

			m_integralImageSurfaceResourceDesc.resType = cudaResourceTypeArray;
			m_integralImageSurfaceResourceDesc.res.array.array = m_integralImagesArray;

			cudaSafeCall(cudaCreateSurfaceObject(&m_integralImageSurface, &m_integralImageSurfaceResourceDesc));
		}


		void ShapeFromSilhouette::freeDeviceMemory()
		{
			cudaSafeCall(cudaFree(m_dev_imagesSegmentation));
			cudaSafeCall(cudaFree(m_dev_integralImages));

			m_dev_imagesSegmentation = nullptr;
			m_dev_integralImages = nullptr;

			cudaSafeCall(cudaDestroySurfaceObject(m_integralImageSurface));
			cudaSafeCall(cudaFreeArray(m_integralImagesArray));

			m_integralImageSurface = 0;
			m_integralImagesArray = nullptr;

			memset(&m_integralImageSurfaceResourceDesc, 0, sizeof(m_integralImageSurfaceResourceDesc));

			m_numImages = 0;
			m_imageSize = { 0, 0 };
		}


		void ShapeFromSilhouette::copyInput(const std::vector<cv::Mat> & imagesSegmentation)
		{
			if (needDeviceInitialization(imagesSegmentation))
			{
				freeDeviceMemory();
				initDeviceMemory(imagesSegmentation);
			}

			const size_t memoryImage = m_imageSize.x * m_imageSize.y;

			unsigned char * dst = m_dev_imagesSegmentation;
			for (size_t i = 0; i < imagesSegmentation.size(); ++i)
			{
				cudaMemcpy(dst, imagesSegmentation[i].data, memoryImage, cudaMemcpyHostToDevice);
				dst += memoryImage;
			}
		}

		void ShapeFromSilhouette::createSpace(const Roi3DF & area, const float3 & voxelSize,
			const CameraSet & cameraModels,
			const std::vector<Face> & walls)
		{
			m_space.reset(new Space(area, voxelSize, cameraModels, walls));
		}
	}
}
