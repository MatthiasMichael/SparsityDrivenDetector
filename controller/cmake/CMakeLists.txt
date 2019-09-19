CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Controller )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( CPLEX )

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )
FIND_PACKAGE( Qt5Sql )

SET( CMAKE_AUTOMOC ON )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../fusion/include
	../../interface/include
	../../shape_from_silhouette_bridge/include
	../../shape_from_silhouette_common/include
	../../sparsity_driven_detector/include
	../../timer/include
	../../unreal_input/include
	../../utility/include
	../../visualization_environment/include
	../../visualization_fusion/include
	../../visualization_shape_from_silhouette/include
	../../visualization_sparsity_driven_detector/include
	${CPLEX_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES_OTHER
	../include/Commander.h
	../include/Configuration.h
	../include/Context.h
	../include/Controller.h
	../include/WindowsConsoleListener.h
	../include/application_helper.h
)

SET( INCLUDE_FILES
	${INCLUDE_FILES_OTHER}
)

SET( SOURCE_FILES_OTHER
	../src/Commander.cpp
	../src/Configuration.cpp
	../src/Context.cpp
	../src/Controller.cpp
	../src/WindowsConsoleListener.cpp
)

SET( SOURCE_FILES
	${SOURCE_FILES_OTHER}
)

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	ApplicationTimer
	Environment
	Fusion
	Interface
	ShapeFromSilhouette_Bridge
	ShapeFromSilhouette_Common
	SparsityDrivenDetector
	UnrealInput
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
	Qt5::Sql
	CONAN_PKG::Geometry
	CONAN_PKG::OsgVisualization
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::boost_program_options
	CONAN_PKG::ffmpeg
	CONAN_PKG::libzip
	CONAN_PKG::opencv
	CONAN_PKG::sqlite3
	debug ${CPLEX_LIBRARY_DEBUG}
	optimized ${CPLEX_LIBRARY_RELEASE}
	debug ${CPLEX_ILOCPLEX_LIBRARY_DEBUG}
	optimized ${CPLEX_ILOCPLEX_LIBRARY_RELEASE}
	debug ${CPLEX_CONCERT_LIBRARY_DEBUG}
	optimized ${CPLEX_CONCERT_LIBRARY_RELEASE}
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )