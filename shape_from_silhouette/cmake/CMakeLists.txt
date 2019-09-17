CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( ShapeFromSilhouette )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Gui )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../shape_from_silhouette_common/include
	../../timer/include
	../../utility/include
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

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
	Qt5::Core
	Qt5::Gui
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::opencv
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms/ShapeFromSilhouette" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )