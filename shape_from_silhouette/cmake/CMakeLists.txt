CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Gui )

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
)

SET( INCLUDE_FILES
	../include/ShapeFromSilhouette.h
	../include/Space.h
	../include/ExtendedVoxel.h
	../include/VoxelPredicates.h
	../include/convexHull_legacy.h
)

SET( SOURCE_FILES
	../src/ShapeFromSilhouette.cpp
	../src/Space.cpp
	../src/ExtendedVoxel.cpp
	../src/convexHull_legacy.cpp
)

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	Qt5::Core
	Qt5::Gui
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )