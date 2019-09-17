CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette_Cuda LANGUAGES CXX CUDA )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

INCLUDE_DIRECTORIES(
	../include 
	../../environment/include
	../../shape_from_silhouette_common/include
	../../timer/include
	../../utility/include
)

SET( INCS
	../include/ShapeFromSilhouetteCuda.h
	../include/deviceStructs.h
	../include/SpaceCuda.h
	../include/VoxelMap.h
	../include/TemporaryDeviceVector.h
)

SET( CUDA_INCS
	../include/geometryPredicates.cuh
	../include/kernel_convexHull.cuh
	../include/kernel_spaceCreation.cuh
	../include/kernel_spaceUpdate.cuh
	../include/kernel_spaceUtils.cuh
	../include/kernel_integralImage.cuh
	../include/kernel_VoxelMap.cuh
	../include/Face.cuh
	../include/VoxelSegmentationInfo.cuh
)

SET( SRCS
	../src/ShapeFromSilhouetteCuda.cpp
	../src/SpaceCuda.cpp
	../src/VoxelMap.cpp
)   

SET( CUDA_SRCS
	../src/kernel_convexHull.cu
	../src/kernel_spaceUpdate.cu
	../src/kernel_spaceCreation.cu
	../src/kernel_integralImage.cu
	../src/kernel_VoxelMap.cu
	../src/Face.cu
	../src/VoxelSegmentationInfo.cu
)

ADD_LIBRARY ( ${PROJECT_NAME} STATIC
	${INCS}
	${SRCS}
	${CUDA_INCS}
	${CUDA_SRCS}
)

TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::opencv
)

SOURCE_GROUP( "Header Files\\Cuda" FILES ${CUDA_INCS} )
SOURCE_GROUP( "Source Files\\Cuda" FILES ${CUDA_SRCS} )


SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette")

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)