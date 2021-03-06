# Source: https://github.com/martinsch/pgmlink

# This module finds cplex.
#
# User can give CPLEX_ROOT_DIR as a hint stored in the cmake cache.
#
# It sets the following variables:
#  CPLEX_FOUND              - Set to false, or undefined, if cplex isn't found.
#  CPLEX_INCLUDE_DIRS       - include directory
#  CPLEX_LIBRARIES          - library files
#  CPLEX_SHARED_LIB         - .dll needed to execute binaries

if(WIN32)
    execute_process(COMMAND cmd /C set CPLEX_STUDIO_DIR OUTPUT_VARIABLE CPLEX_STUDIO_DIR_VAR ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT CPLEX_STUDIO_DIR_VAR)
        MESSAGE(FATAL_ERROR "Unable to find CPLEX: environment variable CPLEX_STUDIO_DIR<VERSION> not set.")
    endif()
    
    # If we have multiple CPLEX Versions installed we get multiple lines. 
    # We try to find the most recent one
    SET(LAST_NEWLINE_IDX -1)
    STRING( FIND ${CPLEX_STUDIO_DIR_VAR} "\n" LAST_NEWLINE_IDX REVERSE)
    IF( NOT ${LAST_NEWLINE_IDX} STREQUAL "-1" )
        STRING(SUBSTRING ${CPLEX_STUDIO_DIR_VAR} ${LAST_NEWLINE_IDX} "-1" CPLEX_STUDIO_DIR_VAR)
        STRING(STRIP ${CPLEX_STUDIO_DIR_VAR} CPLEX_STUDIO_DIR_VAR)
    ENDIF()

    STRING(REGEX REPLACE "^CPLEX_STUDIO_DIR" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
    STRING(REGEX MATCH "^[0-9]+" CPLEX_WIN_VERSION ${CPLEX_STUDIO_DIR_VAR})
    STRING(REGEX REPLACE "^[0-9]+=" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
    
    file(TO_CMAKE_PATH "${CPLEX_STUDIO_DIR_VAR}" CPLEX_ROOT_DIR_GUESS)
    
    # Append 0 if the verison length is 3 (sorry for the hack)
    set(CPLEX_WIN_VERSION_LENGTH 0)
    STRING(LENGTH ${CPLEX_WIN_VERSION} CPLEX_WIN_VERSION_LENGTH)
    IF(${CPLEX_WIN_VERSION_LENGTH} STREQUAL "3")
        SET(CPLEX_WIN_VERSION "${CPLEX_WIN_VERSION}0")
    ENDIF()

    set(CPLEX_WIN_VERSION ${CPLEX_WIN_VERSION} CACHE STRING "CPLEX version to be used.")
    set(CPLEX_ROOT_DIR "${CPLEX_ROOT_DIR_GUESS}" CACHE PATH "CPLEX root directory.")

    #MESSAGE(STATUS "Found CLPEX version ${CPLEX_WIN_VERSION} at '${CPLEX_ROOT_DIR}'")

    STRING(REGEX REPLACE "/VC/bin/.*" "" VISUAL_STUDIO_PATH ${CMAKE_C_COMPILER})
    STRING(REGEX MATCH "Studio [0-9]+" CPLEX_WIN_VS_VERSION ${VISUAL_STUDIO_PATH})
    STRING(REGEX REPLACE "Studio " "" CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION})

    if(${CPLEX_WIN_VS_VERSION} STREQUAL "9")
        set(CPLEX_WIN_VS_VERSION 2008)
    elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "10")
        set(CPLEX_WIN_VS_VERSION 2010)
    elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "11")
        set(CPLEX_WIN_VS_VERSION 2012)
    elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "14")
        set(CPLEX_WIN_VS_VERSION 2015)
    elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "15")
        if(${CPLEX_WIN_VERSION} STREQUAL "1271")
            set(CPLEX_WIN_VS_VERSION 2015)
        else()
            set(CPLEX_WIN_VS_VERSION 2017)
        endif()
    else()
        MESSAGE(FATAL_ERROR "CPLEX: unknown Visual Studio version at '${VISUAL_STUDIO_PATH}'.")
    endif()

    set(CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION} CACHE STRING "Visual Studio Version")
    
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4") 
        set(CPLEX_WIN_BITNESS x86)
    else()
        set(CPLEX_WIN_BITNESS x64)
    endif()

    set(CPLEX_WIN_BITNESS ${CPLEX_WIN_BITNESS} CACHE STRING "On Windows: x86 or x64 (32bit resp. 64bit)")

    #MESSAGE(STATUS "CPLEX: using Visual Studio ${CPLEX_WIN_VS_VERSION} ${CPLEX_WIN_BITNESS} at '${VISUAL_STUDIO_PATH}'")

    # now, generate platform string
    set(CPLEX_WIN_PLATFORM "${CPLEX_WIN_BITNESS}_windows_vs${CPLEX_WIN_VS_VERSION}/stat_")
    
    #MESSAGE("WPF: ${CPLEX_WIN_PLATFORM}")

else()

    set(CPLEX_ROOT_DIR "" CACHE PATH "CPLEX root directory.")
    set(CPLEX_WIN_PLATFORM "")

endif()

FIND_PATH(CPLEX_INCLUDE_DIR
        ilcplex/cplex.h
        HINTS ${CPLEX_ROOT_DIR}/cplex/include
        ${CPLEX_ROOT_DIR}/include
        PATHS ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
        )

FIND_PATH(CPLEX_CONCERT_INCLUDE_DIR
        ilconcert/iloenv.h
        HINTS ${CPLEX_ROOT_DIR}/concert/include
        ${CPLEX_ROOT_DIR}/include
        PATHS ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
        )

FIND_LIBRARY(CPLEX_LIBRARY_RELEASE
        NAMES cplex${CPLEX_WIN_VERSION} cplex
        HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}mda #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH #unix
        ENV LD_LIBRARY_PATH #unix
        )

FIND_LIBRARY(CPLEX_LIBRARY_DEBUG
        NAMES cplex${CPLEX_WIN_VERSION} cplex
        HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}mdd #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH #unix
        ENV LD_LIBRARY_PATH #unix
        )

#message(STATUS "CPLEX Library: ${CPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_ILOCPLEX_LIBRARY_RELEASE
        ilocplex
        HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}mda #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        )
		
FIND_LIBRARY(CPLEX_ILOCPLEX_LIBRARY_DEBUG
        ilocplex
        HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}mdd #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        )

#message(STATUS "ILOCPLEX Library: ${CPLEX_ILOCPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_CONCERT_LIBRARY_RELEASE
        concert
        HINTS ${CPLEX_ROOT_DIR}/concert/lib/${CPLEX_WIN_PLATFORM}mda #windows
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_debian4.0_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        )

FIND_LIBRARY(CPLEX_CONCERT_LIBRARY_DEBUG
        concert
        HINTS ${CPLEX_ROOT_DIR}/concert/lib/${CPLEX_WIN_PLATFORM}mdd #windows
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_debian4.0_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_sles10_4.1/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic #unix 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_osx/static_pic #osx 
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_darwin/static_pic #osx 
        PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        )

#message(STATUS "CONCERT Library: ${CPLEX_CONCERT_LIBRARY}")

if(WIN32)
    FIND_PATH(CPLEX_BIN_DIR
            cplex${CPLEX_WIN_VERSION}.dll
            HINTS ${CPLEX_ROOT_DIR}/cplex/bin/${CPLEX_WIN_PLATFORM} #windows
            )

    SET(CPLEX_SHARED_LIB ${CPLEX_BIN_DIR}/cplex${CPLEX_WIN_VERSION}.dll)
else()
    FIND_PATH(CPLEX_BIN_DIR
            cplex
            HINTS ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_sles10_4.1 #unix
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_debian4.0_4.1 #unix
            ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux #unix
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_osx #osx
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_darwin #osx
            ENV LIBRARY_PATH
            ENV LD_LIBRARY_PATH
            )
endif()

#message(STATUS "CPLEX Bin Dir: ${CPLEX_BIN_DIR}")
#message(STATUS "CPLEX Shared:  ${CPLEX_SHARED_LIB}")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG
        CPLEX_LIBRARY_RELEASE CPLEX_LIBRARY_DEBUG CPLEX_INCLUDE_DIR CPLEX_ILOCPLEX_LIBRARY_DEBUG CPLEX_ILOCPLEX_LIBRARY_RELEASE CPLEX_CONCERT_LIBRARY_RELEASE CPLEX_CONCERT_LIBRARY_DEBUG CPLEX_CONCERT_INCLUDE_DIR)

IF(CPLEX_FOUND)
    SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} ${CPLEX_CONCERT_INCLUDE_DIR})
    SET(CPLEX_LIBRARIES ${CPLEX_CONCERT_LIBRARY_RELEASE} ${CPLEX_ILOCPLEX_LIBRARY_RELEASE} ${CPLEX_LIBRARY_RELEASE} )
    IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
    ENDIF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(CPLEX_LIBRARY_RELEASE CPLEX_LIBRARY_DEBUG CPLEX_INCLUDE_DIR CPLEX_ILOCPLEX_LIBRARY_RELEASE CPLEX_ILOCPLEX_LIBRARY_DEBUG CPLEX_CONCERT_INCLUDE_DIR CPLEX_CONCERT_LIBRARY_RELEASE CPLEX_CONCERT_LIBRARY_DEBUG)