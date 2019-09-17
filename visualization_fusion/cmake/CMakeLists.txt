CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Visualization_Fusion )

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
	../../fusion/include
	../../shape_from_silhouette_common/include
	../../sparsity_driven_detector/include
	../../timer/include
	../../unreal_input/include
	../../utility/include
	../../visualization_environment/include
	../../visualization_shape_from_silhouette/include
	${CPLEX_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES
	../include/DisplayDescriptors_Fusion.h
	../include/DisplayableFusion.h
)

SET( SOURCE_FILES
	../src/DisplayableFusion.cpp
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	Fusion
	ShapeFromSilhouette_Common
	SparsityDrivenDetector
	Visualization_Environment
	Visualization_ShapeFromSilhouette
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
	${CPLEX_LIBRARIES}
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Visualization" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )