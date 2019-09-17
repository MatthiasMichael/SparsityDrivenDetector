CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Fusion )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( CPLEX )

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Gui )

SET( CMAKE_AUTOMOC ON )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../shape_from_silhouette_common/include
	../../sparsity_driven_detector/include
	../../timer/include
	../../unreal_input/include
	../../utility/include
	${CPLEX_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES
	../include/Fusion.h
)

SET( SOURCE_FILES
	../src/Fusion.cpp
)

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	ApplicationTimer
	Environment
	ShapeFromSilhouette_Common
	SparsityDrivenDetector
	UnrealInput
	Qt5::Core
	Qt5::Gui
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::boost_program_options
	CONAN_PKG::ffmpeg
	CONAN_PKG::libzip
	CONAN_PKG::opencv
	CONAN_PKG::sqlite3
	${CPLEX_LIBRARIES}
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )