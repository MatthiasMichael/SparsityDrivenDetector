CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette_Common )

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../utility/include
)

SET( INCLUDE_FILES
	../include/cuda_vector_types_interop.h
	../include/cuda_host_defines_interop.h
	../include/cuda_vector_functions_interop.h
	../include/vector_type_helper.h
	../include/voxel_clustering.h
	../include/space_sequentialFill.h
	../include/Voxel.h
	../include/VoxelCluster.h
)

SET( SOURCE_FILES
	../src/space_sequentialFill.cpp
	../src/voxel_clustering.cpp
	../src/VoxelCluster.cpp
)

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	Environment
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
