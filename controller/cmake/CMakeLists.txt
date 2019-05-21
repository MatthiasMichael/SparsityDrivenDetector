CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Controller )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

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
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Widgets
	Qt5::Sql
	Glu32
	Opengl32
	SparsityDrivenDetector
	Environment
	UnrealInput
	Visualization_Environment
	ApplicationTimer
	ShapeFromSilhouette_Common
	ShapeFromSilhouette_Bridge
	Fusion
	Visualization_ShapeFromSilhouette
	Visualization_SparsityDrivenDetector
	Visualization_Fusion
	Interface
)

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Application" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )