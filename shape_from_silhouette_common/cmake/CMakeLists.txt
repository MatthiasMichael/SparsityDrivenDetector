CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette_Common )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
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
)

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
