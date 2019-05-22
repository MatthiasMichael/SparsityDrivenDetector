CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Visualization_Environment )

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )


INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../utility/include
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES
	../include/DisplayDescriptors_Environment.h
	../include/ActorGeometry.h
	../include/DisplayableFrame.h
	../include/DisplayableMesh.h
)

SET( SOURCE_FILES
	../src/ActorGeometry.cpp
	../src/DisplayableFrame.cpp
	../src/DisplayableMesh.cpp
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
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
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Visualization" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )