CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Executables )

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

INCLUDE_DIRECTORIES(
	../include
	../../controller/include
	../../environment/include
	../../experiment/include
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
	ApplicationTimer
	Controller
	Environment
	Experiment
	Fusion
	Interface
	ShapeFromSilhouette
	ShapeFromSilhouette_Bridge
	ShapeFromSilhouette_Common
	ShapeFromSilhouette_Cuda
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
)

MACRO(ADD_EXE_PROJECT EXE_PROJECT_NAME)
	# Files are passed as optional arguments
	SET (EXE_PROJECT_FILES ${ARGN})

	ADD_EXECUTABLE(${EXE_PROJECT_NAME}
		${EXE_PROJECT_FILES}
	)

	TARGET_LINK_LIBRARIES(${EXE_PROJECT_NAME}
		${COMMON_LIBRARIES}
		debug ${CPLEX_LIBRARY_DEBUG}
		optimized ${CPLEX_LIBRARY_RELEASE}
		debug ${CPLEX_ILOCPLEX_LIBRARY_DEBUG}
		optimized ${CPLEX_ILOCPLEX_LIBRARY_RELEASE}
		debug ${CPLEX_CONCERT_LIBRARY_DEBUG}
		optimized ${CPLEX_CONCERT_LIBRARY_RELEASE}
	)

	CONAN_TARGET_LINK_LIBRARIES(${EXE_PROJECT_NAME})

	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY FOLDER "Executables" )
	
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
	
	SET_PROPERTY( TARGET ${EXE_PROJECT_NAME} PROPERTY LINK_FLAGS "/ignore:4099,4075" )
	
	GET_TARGET_PROPERTY(QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
	GET_FILENAME_COMPONENT(QT_BIN_DIR "${QMAKE_EXECUTABLE}" DIRECTORY)
	FIND_PROGRAM(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")

	ADD_CUSTOM_COMMAND(TARGET ${EXE_PROJECT_NAME} POST_BUILD
		COMMAND "${CMAKE_COMMAND}" -E
		env PATH="${_qt_bin_dir}" "${WINDEPLOYQT_EXECUTABLE}" "$<TARGET_FILE:${EXE_PROJECT_NAME}>" "--dir" "${CMAKE_CURRENT_BINARY_DIR}"
		COMMENT "Running windeployqt..."
	)
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

FILE(
    COPY ${CPLEX_SHARED_LIB}
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
