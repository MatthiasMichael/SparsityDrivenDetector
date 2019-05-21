CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( Environment )

INCLUDE(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
CONAN_BASIC_SETUP()

MESSAGE("Flags ${PROJECT_NAME}:")
SHOW_CMAKE_CXX_FLAGS()

FIND_PACKAGE( Qt5Core )

message("INC: ${LIB_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(
	${LIB_INCLUDE_DIRS}
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

CONAN_TARGET_LINK_LIBRARIES(${PROJECT_NAME})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	Qt5::Core
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Input" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )