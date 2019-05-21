CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Visualization_SparsityDrivenDetector )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )

INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
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

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	Qt5::Core
	Qt5::Widgets
	Qt5::Gui
	Qt5::OpenGL
	Glu32
	Opengl32
	Environment
	ApplicationTimer
	SparsityDrivenDetector
	Visualization_Environment
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Visualization" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )