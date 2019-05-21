CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Utility )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
)

ADD_CUSTOM_TARGET( ${PROJECT_NAME} SOURCES
	../include/listFilesInDirectory.h
	../include/TemporaryJoinedVector.h
	../include/TemporaryVector.h
	../include/cuda_analyticGeometry.h
	../include/cuda_error_check.h
	../include/cuda_host_device.h
	../include/cuda_math_utils.h
	../include/enumerate.h
	../include/vectorOperations.h
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Utility" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )