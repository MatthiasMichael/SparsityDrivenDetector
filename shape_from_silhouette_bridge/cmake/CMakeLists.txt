CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette_Bridge LANGUAGES CXX CUDA )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

INCLUDE_DIRECTORIES(
	../include 
	../../environment/include
	../../shape_from_silhouette/include
	../../shape_from_silhouette_common/include
	../../shape_from_silhouette_cuda/include
	../../timer/include
	../../utility/include
)

SET( INCS
	../include/ShapeFromSilhouetteBridge.h
	../include/ShapeFromSilhouette_Impl.h
	../include/ShapeFromSilhouette_ImplCuda.h
	../include/ShapeFromSilhouette_ImplHost.h
)

SET( SRCS
	../src/ShapeFromSilhouetteBridge.cpp
	../src/ShapeFromSilhouette_ImplCuda.cpp
	../src/ShapeFromSilhouette_ImplHost.cpp
)   

ADD_LIBRARY ( ${PROJECT_NAME} STATIC
	${INCS}
	${SRCS}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette
	ShapeFromSilhouette_Common
	ShapeFromSilhouette_Cuda
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::opencv
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette")

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)