CMAKE_MINIMUM_REQUIRED( VERSION 3.10 ) # For set property CXX_STANDARD 17

PROJECT( SparsityDrivenDetector )

IF( ${VERBOSE_CONFIGURE} )
	MESSAGE("Flags ${PROJECT_NAME}:")
	SHOW_CMAKE_CXX_FLAGS()
ENDIF()

FIND_PACKAGE( CPLEX )
FIND_PACKAGE( Qt5Core )

INCLUDE_DIRECTORIES(
	../include
	../../environment/include
	../../timer/include
	../../utility/include
	${CPLEX_INCLUDE_DIRS}
)

SET( INCLUDE_FILES_OTHER
	../include/Solution.h
	../include/SparsityDrivenDetector.h
	../include/SparsityDrivenDetectorPostProcessing.h
	../include/SparseMatrix.h
	../include/conversion_helper.h
)

SET( INCLUDE_FILES_DICTIONARY
	../include/Dictionary.h
	../include/GridPoints.h
	../include/Template.h
	../include/TemplateTransformer.h
)

SET( INCLUDE_FILES_OPTIM
	../include/IOptimizationProblem.h
	../include/OptimizationProblem.h
	../include/OptimizationProblem_Single.h
	../include/OptimizationProblem_SingleLayered.h
	../include/OptimizationProblem_Multi.h
	../include/OptimizationProblem_MultiLayered.h
)

SET( INCLUDE_FILES_OPTIM_FACTORY
	../include/OptimizationProblemFactory.h
	../include/OptimizationProblemFactory_Single.h
	../include/OptimizationProblemFactory_SingleLayered.h
	../include/OptimizationProblemFactory_Multi.h
	../include/OptimizationProblemFactory_MultiLayered.h
	../include/OptimizationProblemFactoryCollection.h
)

SET( INCLUDE_FILES_OPTIM_DECORATOR
	../include/OptimizationProblemDecorator.h
	../include/OptimizationProblemWriter.h
	../include/OptimizationProblemMatrixWriter.h
	../include/OptimizationProblemTimer.h
)

SET( INCLUDE_FILES
	${INCLUDE_FILES_OTHER}
	${INCLUDE_FILES_OPTIM}
	${INCLUDE_FILES_OPTIM_FACTORY}
	${INCLUDE_FILES_OPTIM_DECORATOR}
	${INCLUDE_FILES_DICTIONARY}
)

SET( SOURCE_FILES_OTHER
	../src/Solution.cpp
	../src/SparsityDrivenDetector.cpp
	../src/SparsityDrivenDetectorPostProcessing.cpp
	../src/conversion_helper.cpp
)

SET( SOURCE_FILES_DICTIONARY
	../src/Dictionary.cpp
	../src/GridPoints.cpp
	../src/Template.cpp
	../src/TemplateTransformer.cpp
)

SET( SOURCE_FILES_OPTIM
	../src/OptimizationProblem.cpp
	../src/OptimizationProblem_Single.cpp
	../src/OptimizationProblem_SingleLayered.cpp
	../src/OptimizationProblem_Multi.cpp
	../src/OptimizationProblem_MultiLayered.cpp
)

SET( SOURCE_FILES_OPTIM_FACTORY
	../src/OptimizationProblemFactory.cpp
	../src/OptimizationProblemFactory_Single.cpp
	../src/OptimizationProblemFactory_SingleLayered.cpp
	../src/OptimizationProblemFactory_Multi.cpp
	../src/OptimizationProblemFactory_MultiLayered.cpp
)

SET( SOURCE_FILES_OPTIM_DECORATOR
	../src/OptimizationProblemDecorator.cpp
	../src/OptimizationProblemWriter.cpp
	../src/OptimizationProblemMatrixWriter.cpp
	../src/OptimizationProblemTimer.cpp
)

SET( SOURCE_FILES
	${SOURCE_FILES_OTHER}
	${SOURCE_FILES_OPTIM}
	${SOURCE_FILES_OPTIM_FACTORY}
	${SOURCE_FILES_OPTIM_DECORATOR}
	${SOURCE_FILES_DICTIONARY}
)

ADD_LIBRARY( ${PROJECT_NAME} STATIC
	${INCLUDE_FILES}
	${SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	ApplicationTimer
	Environment
	Qt5::Core
	CONAN_PKG::Geometry
	CONAN_PKG::Roi3DF
	CONAN_PKG::boost_filesystem
	CONAN_PKG::libzip
	CONAN_PKG::opencv
	${CPLEX_LIBRARIES}
)

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY FOLDER "Algorithms" )

SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
SET_PROPERTY( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )

SOURCE_GROUP( "Header Files\\Dictionary" FILES ${INCLUDE_FILES_DICTIONARY} )
SOURCE_GROUP( "Header Files\\OptimizationProblem" FILES ${INCLUDE_FILES_OPTIM} )
SOURCE_GROUP( "Header Files\\OptimizationProblemFactory" FILES ${INCLUDE_FILES_OPTIM_FACTORY} )
SOURCE_GROUP( "Header Files\\OptimizationProblemDecorator" FILES ${INCLUDE_FILES_OPTIM_DECORATOR} )

SOURCE_GROUP( "Source Files\\Dictionary" FILES ${SOURCE_FILES_DICTIONARY} )
SOURCE_GROUP( "Source Files\\OptimizationProblem" FILES ${SOURCE_FILES_OPTIM} )
SOURCE_GROUP( "Source Files\\OptimizationProblemFactory" FILES ${SOURCE_FILES_OPTIM_FACTORY} )
SOURCE_GROUP( "Source Files\\OptimizationProblemDecorator" FILES ${SOURCE_FILES_OPTIM_DECORATOR} )