CMAKE_MINIMUM_REQUIRED( VERSION 3.10 )

PROJECT( SparsityDrivenDetector )

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") 

INCLUDE(conan.cmake)
conan_cmake_run(
	CONANFILE ../conanfile.py
	BASIC_SETUP CMAKE_TARGETS
	GENERATORS cmake_multi
	BUILD missing)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

ADD_DEFINITIONS(-DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE)
ADD_DEFINITIONS(-D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING)

SET(VERBOSE_CONFIGURE OFF CACHE BOOL "Show additional information during the configure step")

MACRO( SHOW_CMAKE_CXX_FLAGS )
	MESSAGE("Flags Release:        ${CMAKE_CXX_FLAGS_RELEASE}")
	MESSAGE("Flags RelWithDebInfo: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
	MESSAGE("Flags Debug:          ${CMAKE_CXX_FLAGS_DEBUG}")
ENDMACRO()

IF(${VERBOSE_CONFIGURE})
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

SET( LIB_PROJECTS
	controller
	environment
	experiment
	fusion
	interface
	shape_from_silhouette
	shape_from_silhouette_bridge
	shape_from_silhouette_common
	shape_from_silhouette_cuda
	sparsity_driven_detector
	timer
	unreal_input
	utility
	visualization_environment
	visualization_fusion
	visualization_shape_from_silhouette
	visualization_sparsity_driven_detector
)

FOREACH(LIB_PROJECT IN LISTS LIB_PROJECTS)
	ADD_SUBDIRECTORY( ../${LIB_PROJECT}/cmake ${CMAKE_CURRENT_BINARY_DIR}/${LIB_PROJECT} )
ENDFOREACH()

ADD_SUBDIRECTORY( ../executable/cmake ${CMAKE_CURRENT_BINARY_DIR}/executable )

ADD_SUBDIRECTORY( ../analysis/cmake ${CMAKE_CURRENT_BINARY_DIR}/analysis )