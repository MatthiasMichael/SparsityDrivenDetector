CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Environment )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( Qt5Core )

INCLUDE_DIRECTORIES(
	../include
)

SET( INCLUDE_FILES
	../include/Actor.h
	../include/Environment.h
	../include/Frame.h
	../include/Face.h
	../include/Mesh.h
	../include/CommonGeometricTypes.h
	../include/WorldCoordinateSystem_SDD.h
	../include/IdentifiableCamera.h
	../include/serialization_helper.h
)

SET( SOURCE_FILES
	../src/Actor.cpp
	../src/Environment.cpp
	../src/Frame.cpp
	../src/Face.cpp
	../src/Mesh.cpp
	../src/IdentifiableCamera.cpp
	../src/serialization_helper.cpp
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	Qt5::Core
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Input" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )