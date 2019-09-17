CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Visualization_ShapeFromSilhouette )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../shape_from_silhouette_common/include
	../../utility/include
)

SET( INCLUDE_FILES_OTHER
	../include/DisplayDescriptors_ShapeFromSilhouette.h
)

SET( INCLUDE_FILES_DISPLAYABLE
	../include/DisplayableObjects.h
)

SET( INCLUDE_FILES_GEOMETRY
	../include/VoxelClusterGeometry.h
	../include/VoxelGeometry.h
)

SET( INCLUDE_FILES
	${INCLUDE_FILES_DISPLAYABLE}
	${INCLUDE_FILES_GEOMETRY}
	${INCLUDE_FILES_OTHER}
)

SET( SOURCE_FILES_DISPLAYABLE
	../src/DisplayableObjects.cpp
)

SET( SOURCE_FILES_GEOMETRY
	../src/VoxelClusterGeometry.cpp
	../src/VoxelGeometry.cpp
)

SET( SOURCE_FILES
	${SOURCE_FILES_DISPLAYABLE}
	${SOURCE_FILES_GEOMETRY}
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
	Glu32
	Opengl32
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Widgets
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::OsgVisualization
	CONAN_PKG::opencv
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Visualization" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )

SOURCE_GROUP( "Header Files\\Displayable" FILES ${INCLUDE_FILES_DISPLAYABLE} )
SOURCE_GROUP( "Header Files\\Geometry" FILES ${INCLUDE_FILES_GEOMETRY} )

SOURCE_GROUP( "Source Files\\Displayable" FILES ${SOURCE_FILES_DISPLAYABLE} )
SOURCE_GROUP( "Source Files\\Geometry" FILES ${SOURCE_FILES_GEOMETRY} )