CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Interface )

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )

SET( CMAKE_AUTOMOC ON )
SET( CMAKE_AUTOUIC ON )

SET( CMAKE_AUTOUIC_SEARCH_PATHS ../ui )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../fusion/include
	../../shape_from_silhouette_bridge/include
	../../shape_from_silhouette_common/include
	../../sparsity_driven_detector/include
	../../timer/include
	../../utility/include
	../../visualization_environment/include
	../../visualization_fusion/include
	../../visualization_shape_from_silhouette/include
	../../visualization_sparsity_driven_detector/include
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES
	../include/MainWindow.h
)

SET( SOURCE_FILES
	../src/MainWindow.cpp
)

SET( UI_FILES
	../ui/MainWindow.ui
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
	${UI_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	Fusion
	ShapeFromSilhouette_Bridge
	ShapeFromSilhouette_Common
	SparsityDrivenDetector
	Visualization_Environment
	Visualization_Fusion
	Visualization_ShapeFromSilhouette
	Visualization_SparsityDrivenDetector
	Glu32
	Opengl32
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Widgets
	CONAN_PKG::CPLEX
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::ffmpeg
	CONAN_PKG::libzip
	CONAN_PKG::opencv
	CONAN_PKG::sqlite3
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )