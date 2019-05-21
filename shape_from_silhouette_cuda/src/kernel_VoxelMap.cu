#include "kernel_VoxelMap.cuh"

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_analyticGeometry.h"
#include "cuda_error_check.h"
#include "cuda_math_utils.h"


namespace sfs
{
	namespace cuda
	{

#define MAX_NUM_CLUSTER 254 // Index Type is uchar and 0 is a marker for "no singular view ray"
#define MAX_NUM_FACES 256 // TODO: Multiple kernel passes when this number is exceeded. Maybe not using constant memory is faster then
#define MAX_NUM_CAMERAS 16 // TODO: An zentralerer Stelle festlegen?


		__constant__ DeviceVoxelCluster const_deviceCluster[MAX_NUM_CLUSTER];
		__constant__ Face const_deviceFaces[MAX_NUM_FACES];
		__constant__ float3 const_deviceCameraCenters[MAX_NUM_CAMERAS];


		__host__ __device__ bool intersectsInPlane(const Roi3DF & roi, const DeviceViewRay & ray)
		{
			float3 points[4];
			float3 directions[4];

			points[0] = make_float3(roi.x1, roi.y1, 0);
			points[1] = make_float3(roi.x1, roi.y1, 0);
			points[2] = make_float3(roi.x2, roi.y2, 0);
			points[3] = make_float3(roi.x2, roi.y2, 0);

			directions[0] = make_float3(roi.x2 - roi.x1, roi.y1 - roi.y1, 0);
			directions[1] = make_float3(roi.x1 - roi.x1, roi.y2 - roi.y1, 0);
			directions[2] = make_float3(roi.x2 - roi.x2, roi.y1 - roi.y2, 0);
			directions[3] = make_float3(roi.x1 - roi.x2, roi.y2 - roi.y2, 0);

			const float3 rayOrigin = make_float3(ray.origin.x, ray.origin.y, 0);
			const float3 rayDirection = make_float3(ray.ray.x, ray.ray.y, 0);

			//#pragma unroll 4
			for (int i = 0; i < 4; ++i)
			{
				float3 closestPoint;
				float scalar1, scalar2;

				intersectLineAndLine(&closestPoint, rayOrigin, rayDirection, points[i], directions[i], &scalar1, &scalar2);

				if (scalar2 >= 0 && scalar2 <= norm(directions[i])) // scalar2 is for x = p2 + scala2 * (v2 / v2.norm())
				{
					return true;
				}
			}
			return false;
		}


		__device__ inline bool isInsideImage(uint x, uint y, uint2 imgSize)
		{
			return x < imgSize.x && y < imgSize.y;
		}


		__device__ inline bool isValidPixelMapOffset(uint idxVoxel, uint idxCorner, uint idxCamera, uint3 numVoxel, uint numCorners, uint numCameras)
		{
			return idxVoxel < prod(numVoxel) && idxCorner < numCorners && idxCamera < numCameras;
		}


		__global__ void checkForSingularViewRays(DeviceViewRay * pViewRays, unsigned char * pImagesSegmentation, unsigned char * viewRayMaps, uint2 imgSize, uint numImages, uint numCluster)
		{
			const uint idxX = blockIdx.x * blockDim.x + threadIdx.x;
			const uint idxY = blockIdx.y * blockDim.y + threadIdx.y;
			const uint idxCam = blockIdx.z;

			/*
			* Erzeugt ein Ausgabebild wo jeder Pixel markiert ist mit dem Index des Clusters den der Sichtstrahl einzigartig schneidet.
			* -> Wir brauchen auch eine HostVoxelMap um die Pixel dann den Sichtstrahlen zuzuordnen
			*/

			if (isInsideImage(idxX, idxY, imgSize))
			{
				const uint offsetImage = idxCam * imgSize.x * imgSize.y;
				const uint offsetPixel = idxY * imgSize.x + idxX;

				uint counterIntersections = 0;

				for (uint idxCluster = 0; idxCluster < numCluster; ++idxCluster)
				{
					const bool intersectsCluster = intersectsInPlane(const_deviceCluster[idxCluster].boundingBox, pViewRays[offsetImage + offsetPixel]);
					if (intersectsCluster && pImagesSegmentation[offsetImage + offsetPixel] != 0)
					{
						++counterIntersections;
						if (counterIntersections == 1)
						{
							viewRayMaps[offsetImage + offsetPixel] = idxCluster + 1; // 0 means no cluster therefore we have to start at 1
						}
					}
				}
			}
		}


		__global__ void removePixelMapEntriesToNonVisiblePoints(int2 * pPixelMaps, Voxel * pVoxel, uint3 numVoxel, uint numCorners, uint numCameras, uint numFaces)
		{
			const uint idxVoxel = blockIdx.x * blockDim.x + threadIdx.x;
			const uint idxCorner = blockIdx.y * blockDim.y + threadIdx.y;
			const uint idxCamera = blockIdx.z;

			const uint pixelMapOffset = idxVoxel * numCorners * numCameras + idxCorner * numCameras + idxCamera;

			if (isValidPixelMapOffset(idxVoxel, idxCorner, idxCamera, numVoxel, numCorners, numCameras))
			{
				const float3 & cameraCenter = const_deviceCameraCenters[idxCamera];
				const float3 & voxelCorner = pVoxel[idxVoxel].getCorners()[idxCorner];

				bool occluded = false;
				for (uint idxFace = 0; idxFace < numFaces; ++idxFace)
				{
					const Face & face = const_deviceFaces[idxFace];
					if (face.isOccluded(voxelCorner, cameraCenter))
					{
						occluded = true;
					}
				}

				if (occluded && pPixelMaps[pixelMapOffset].x != INT_MIN && pPixelMaps[pixelMapOffset].y != INT_MIN)
				{
					pPixelMaps[pixelMapOffset].x = -pPixelMaps[pixelMapOffset].x; // A negative value != INT_MIN means occluded
					pPixelMaps[pixelMapOffset].y = -pPixelMaps[pixelMapOffset].y;
				}
			}
		}


		void call_checkForSingularViewRays(const std::vector<DeviceVoxelCluster> & voxelCluster, DeviceViewRay * p_dev_viewRays, unsigned char * p_dev_imagesSegmentation, unsigned char * p_dev_viewRayMaps, uint numImages, uint2 imageSize)
		{
			const uint numCluster = static_cast<uint>(voxelCluster.size());

			assert(numCluster < MAX_NUM_CLUSTER);
			if (numCluster > MAX_NUM_CLUSTER)
			{
				throw std::runtime_error("Too many clusters. Are you sure the previous processing works correctly?");
			}

			cudaSafeCall(cudaMemcpyToSymbol(const_deviceCluster, voxelCluster.data(), numCluster * sizeof(DeviceVoxelCluster)));

			dim3 blockSize(32, 32);
			dim3 gridSize((imageSize.x + 31) / 32, (imageSize.y + 31) / 32, numImages);

			checkForSingularViewRays << <gridSize, blockSize >> > (p_dev_viewRays, p_dev_imagesSegmentation, p_dev_viewRayMaps, imageSize, numImages, numCluster);

			cudaCheckError();
		}


		void call_removePixelMapEntriesToNonVisiblePoints(int2 * p_dev_pixelMap, Voxel * p_dev_voxel, const std::vector<Face> & faces, std::vector<float3> cameraCenters, uint3 numVoxel)
		{
			const uint numCameras = static_cast<uint>(cameraCenters.size());
			
			if (numCameras > MAX_NUM_CAMERAS)
			{
				throw std::runtime_error("Too much cameras.");
			}

			cudaSafeCall(cudaMemcpyToSymbol(const_deviceCameraCenters, cameraCenters.data(), numCameras * sizeof(float3)));

			dim3 blockSize(128, 8); // Voxels have 8 Corners. I don't think this will change anytime soon.
			dim3 gridSize((prod(numVoxel) + 127) / 128, 1, numCameras);

			uint numFaces = static_cast<uint>(faces.size());
			size_t batchOffset = 0;

			std::cout << "Calculating Voxel occlusions with " << numFaces << " Faces in batches of " << MAX_NUM_FACES << "." << std::endl;
			while (numFaces > 0)
			{
				const uint batchSize = numFaces > MAX_NUM_FACES ? MAX_NUM_FACES : numFaces;

				cudaSafeCall(cudaMemcpyToSymbol(const_deviceFaces, faces.data() + batchOffset, batchSize * sizeof(Face)));

				removePixelMapEntriesToNonVisiblePoints << <gridSize, blockSize >> > (p_dev_pixelMap, p_dev_voxel, numVoxel, 8, numCameras, batchSize);
				
				auto e = cudaDeviceSynchronize();
				if( e != cudaSuccess)
				{
					std::cout << "\nError on launch:\n" << cudaGetErrorString(e);
				}

				cudaCheckError();

				numFaces -= batchSize;
				batchOffset += batchSize;
				std::cout << "\r" << batchOffset << " | " << numFaces << " done.           ";
			}
			std::cout << "\rDone                                                         " << std::endl;
		}

	}
}
