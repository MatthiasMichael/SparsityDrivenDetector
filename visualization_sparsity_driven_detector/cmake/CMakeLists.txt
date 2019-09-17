CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Visualization_SparsityDrivenDetector )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( CPLEX )

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../sparsity_driven_detector/include
	../../timer/include
	../../utility/include
	../../visualization_environment/include
	${CPLEX_INCLUDE_DIRS}
)

SET( INCLUDE_FILES
	../include/DisplayDescriptors_SparsityDrivenDetector.h
	../include/DisplayableGridPoints.h
	../include/DisplayableSolution.h
)

SET( SOURCE_FILES
	../src/DisplayableGridPoints.cpp
	../src/DisplayableSolution.cpp
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)


TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	SparsityDrivenDetector
	Visualization_Environment
	Glu32
	Opengl32
	Qt5::Core
	Qt5::Widgets
	Qt5::Gui
	Qt5::OpenGL
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::OsgVisualization
	CONAN_PKG::opencv
	${CPLEX_LIBRARIES}
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Visualization" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )