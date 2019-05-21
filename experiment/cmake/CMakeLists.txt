CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Experiment )

INCLUDE( $ENV{RTCV_CMAKE_MACROS_DIR}/cmakeMacros.txt )

CONAN_SETUP()
MODIFY_COMPILER_FLAGS()
SET_CMAKE_POLICIES()
SET_RTCV_PATH_VARIABLES()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )
FIND_PACKAGE( Qt5Sql )

SET( CMAKE_AUTOMOC ON )

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( INCLUDE_FILES_EXPERIMENT
	../include/Experiment.h
	../include/ExperimentCommander.h
)

SET( INCLUDE_FILES_EXPORTER
	../include/ConfigurationExporter.h
	../include/ExperimentExporter.h
	../include/TimingExporter.h
)

SET( INCLUDE_FILES
	${INCLUDE_FILES_EXPERIMENT}
	${INCLUDE_FILES_EXPORTER}
)

SET( SOURCE_FILES_EXPERIMENT
	../src/Experiment.cpp
	../src/ExperimentCommander.cpp
)

SET( SOURCE_FILES_EXPORTER
	../src/ConfigurationExporter.cpp
	../src/ExperimentExporter.cpp
	../src/TimingExporter.cpp
)

SET( SOURCE_FILES
	${SOURCE_FILES_EXPERIMENT}
	${SOURCE_FILES_EXPORTER}
)

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Widgets
	Qt5::Sql
	Glu32
	Opengl32
	SparsityDrivenDetector
	UnrealInput
	Environment
	Fusion
	Visualization_Fusion
	Visualization_Environment
	Visualization_SparsityDrivenDetector
	Controller
	ApplicationTimer
	ShapeFromSilhouette_Common
	ShapeFromSilhouette_Bridge
	Visualization_ShapeFromSilhouette
	Interface
)

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )

SOURCE_GROUP( "Header Files\\Exporter" FILES ${INCLUDE_FILES_EXPORTER} )
SOURCE_GROUP( "Source Files\\Exporter" FILES ${SOURCE_FILES_EXPORTER} )

AUTOGEN_TO_FOLDER()