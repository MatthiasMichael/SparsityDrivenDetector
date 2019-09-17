CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT( UnrealInput )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( Qt5Core )
FIND_PACKAGE( Qt5Sql )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../timer/include
	../../utility/include
)

SET( INCLUDE_FILES_OTHER
	../include/ImageSequenceReader.h
	../include/MpegReader.h
	../include/Scene.h
	../include/SceneInfo.h
	../include/VideoInput.h
	../include/VideoInput_Mpeg.h
	../include/VideoInput_ImageSequence.h
	../include/environmentFromSceneInfo.h
)

SET( INCLUDE_FILES_COORDINATE_SYSTEMS
	../include/UnrealCoordinateSystem.h
)

SET( INCLUDE_FILES_SERIALIZATION
	../include/ActorPositionSqlSerializer.h
	../include/ActorTypeSqlSerializer.h
	../include/CameraSqlSerializer.h
	../include/MapSqlSerializer.h
	../include/NavMeshSqlSerializer.h
	../include/SqlSerializer.h
)

SET( INCLUDE_FILES
	${INCLUDE_FILES_OTHER}
	${INCLUDE_FILES_COORDINATE_SYSTEMS}
	${INCLUDE_FILES_SERIALIZATION}
	${INCLUDE_FILES_VISUALIZATION}
	${INCLUDE_FILES_WIDGETS}
)

SET( SOURCE_FILES_OTHER
	../src/ImageSequenceReader.cpp
	../src/MpegReader.cpp
	../src/Scene.cpp
	../src/SceneInfo.cpp
	../src/VideoInput.cpp
	../src/VideoInput_Mpeg.cpp
	../src/VideoInput_ImageSequence.cpp
	../src/environmentFromSceneInfo.cpp
)

SET( SOURCE_FILES_SERIALIZATION
	../src/ActorPositionSqlSerializer.cpp
	../src/ActorTypeSqlSerializer.cpp
	../src/CameraSqlSerializer.cpp
	../src/MapSqlSerializer.cpp
	../src/NavMeshSqlSerializer.cpp
	../src/SqlSerializer.cpp
)

SET( SOURCE_FILES
	${SOURCE_FILES_OTHER}
	${SOURCE_FILES_SERIALIZATION}
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	ApplicationTimer
	Environment
	Qt5::Core
	Qt5::Sql
	CONAN_PKG::opencv
	CONAN_PKG::ffmpeg
	CONAN_PKG::Geometry
	CONAN_PKG::sqlite3
)

SOURCE_GROUP( "Header Files\\CoordinateSystems" FILES ${INCLUDE_FILES_COORDINATE_SYSTEMS} )
SOURCE_GROUP( "Header Files\\Serialization" FILES ${INCLUDE_FILES_SERIALIZATION} )

SOURCE_GROUP( "Source Files\\Serialization" FILES ${SOURCE_FILES_SERIALIZATION} )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Input" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
