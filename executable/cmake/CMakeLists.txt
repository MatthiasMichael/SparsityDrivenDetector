CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Executables )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Widgets )
FIND_PACKAGE( Qt5Gui )
FIND_PACKAGE( Qt5OpenGL )
FIND_PACKAGE( Qt5Sql )

INCLUDE_DIRECTORIES(
	../include
	${LIB_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
)

SET( RESOURCE_FOLDER
	../res
)

SET( RESOURCE_FILENAMES
	person_01.ppm
	person_02.ppm
)

# Build the complete path for each resource file
SET( RESOURCE_FILES "")
FOREACH(file ${RESOURCE_FILENAMES})
	LIST(APPEND RESOURCE_FILES "${RESOURCE_FOLDER}/${file}")
ENDFOREACH()

SET( COMMON_FILES
	${RESOURCE_FILES}
)

SET( COMMON_LIBRARIES
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
	Experiment
	ApplicationTimer
	ShapeFromSilhouette_Common
	ShapeFromSilhouette_Bridge
	ShapeFromSilhouette
	ShapeFromSilhouette_Cuda
	Visualization_ShapeFromSilhouette
	Interface
)

MACRO(ADD_EXE_PROJECT EXE_PROJECT_NAME)
	# Files are passed as optional arguments
	SET (EXE_PROJECT_FILES ${ARGN})

	ADD_EXECUTABLE(${EXE_PROJECT_NAME}
		${EXE_PROJECT_FILES}
	)

	TARGET_LINK_LIBRARIES(${EXE_PROJECT_NAME}
		${COMMON_LIBRARIES}
	)

	CONAN_TARGET_LINK_LIBRARIES(${EXE_PROJECT_NAME})

	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY FOLDER "Executables" )
	
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
	
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY LINK_FLAGS "/ignore:4099,4075" )
ENDMACRO(ADD_EXE_PROJECT)

ADD_EXE_PROJECT( SDD_Interactive
	../src/interactive.cpp
)

ADD_EXE_PROJECT( SDD_Experiments
	../src/experiments.cpp
)

# These Projects were for Matlab interoperability. Currently the C++ Version is too far ahead for these to work
#ADD_EXE_PROJECT( SDD_UEScene_Create
#	../src/SDD_UEScene_Create.cpp
#)
#
#ADD_EXE_PROJECT( SDD_UEScene_LoadDict
#	../src/SDD_UEScene_LoadDict.cpp
#)
#
#ADD_EXE_PROJECT( SDD_UEScene_LoadDetector
#	../src/SDD_UEScene_LoadDetector.cpp
#)

# Copy each resouce file in the folder res in the current binary dir
FOREACH(item IN LISTS RESOURCE_FILENAMES)
    ADD_CUSTOM_COMMAND(
        TARGET SDD_Interactive POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_FOLDER}/${item}" "${CMAKE_CURRENT_BINARY_DIR}/res/${item}"
    )
ENDFOREACH()

