CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette_Bridge LANGUAGES CXX CUDA )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
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

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
	ShapeFromSilhouette
	ShapeFromSilhouette_Cuda
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette")

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)