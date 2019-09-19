# SparsityDrivenDetector
This is code provided for the paper *Fusing Shape-from-Silhouette and the Sparsity Driven Detector for Camera-Based 3D Multi-Object Localization with Occlusions*.
Currently it is only tested on Windows with Visual Studio 2017. In order to use a different compiler several aspects of the build system which is described further down will have to be tweaked. Therefore the descriptions below are targeted to building on this specific platform.

## How to Build
Dependency management for C++ on Windows always has been a major issue. To make things easier this project uses [conan](https://conan.io) for dependency management and CMake as build system.

### Conan Dependencies
Install conan as described on the official website.  
In addition to the official `conan-center`repository you will also need the `bincrafters` repository (for libraries like `boost` and `ffmpeg`) which can be added with:

    conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
    
The majority of dependencies will automatically be downloaded and installed when you invoke CMake.
At the time of writing however, OpenSceneGraph can not be found in any of the official repositories. But there exists a corresponding recipe in https://github.com/bincrafters/conan-openscenegraph. To make this package usable on youre machine you have to execute:

    git clone https://github.com/bincrafters/conan-openscenegraph.git
    cd conan-openscenegraph
    conan create . bincrafters/stable
    
### Non-Conan Dependencies
There are other libraries that cannot be found in the remotes and which need to be installed manually. These are:
* Qt5
* CUDA 10
* CPLEX >=12.7.1 (https://www.ibm.com/analytics/cplex-optimizer)

CPLEX is proprietary software but can be downloaded and used freely for academic and research purposes. When you install it it automatically sets an environment variable called `CPLEX_STUDIO_DIR<version>`. The CMake script tries to find the latest of these entries and infers the relevant parts of the library from that.

Support for CUDA and Qt is present in standard CMake. You might have to specify the `CMAKE_PREFIX_PATH` when invoking CMake.
    
### Dependencies to my own libraries
At last this project depends on five other libraries that can be found in other repositories of this account:
* NamedType
* Geometry
* Parametrizable
* OsgVisualization
* Roi3DF

Since these are in part interdependent `conan` is used to hide most of the complexity.

## Building
You have two ways of building executables of the Sparsity Driven Detector. You can either use the simple method of using conan workspaces to build all of our code at once. (This is labled as experimental by the developers of conan but has worked well for me so far.) Or you can make the five libraries from the previous section available als local conan packages.

### As a Single Workspace
Check out the code in https://github.com/MatthiasMichael/conanws-SparsityDrivenDetector:

    git clone https://github.com/MatthiasMichael/conanws-SparsityDrivenDetector.git
    git submodule init
    git submodule update --recursive --remote
    
Invoke CMake in your preferred way -- either via the `cmake-gui` or the command line.

### As Individual Packages 
To make each package of the list NamedType, Geometry, Parametrizable, OsgVisualization, Roi3DF available you have to run:

    git clone https://github.com/MatthiasMichael/<package>.git
    cd <package>
    conan export . MatthiasMichael/stable
    conan install <package>/1.0@MatthiasMichael/stable -- build missing
    
Then clone this repository and invoke CMake in your preferred way.

## Running
If you did everything corectly you end up with two executables: `SDD_Interactive` and `SDD_Experiment`.

### SDD_Interactive
This executable allows you to play around and test the available functionality of our system. When run it expects the path to a settings file as an argument:

    SDD_Interactive.exe -c path/to/settings.ini
    
Example data for this can be found in the `res` folder (which is automatically copied to the build directory after building): A scene-file, two templates and a `settings.ini`. 
If no settings file is provided, SDD_Interactive defaults to `res/settings.ini`. So if you run it from Visual Studio it should find the default settings.

From those settings it builds the optimization problem and -- once finished -- shows a simple 3D rendering of the scene.
Typing `h` in the command line displays a list of available commands. 
`r` lets the scene run in endless mode.

### SDD_Experiment
TODO...

## Known Issues
Currenty the build process is tuned towards Windows and Visual Studio. Conan is explicitly invoked with the cmake-multi generator to make switching between Release and Debug builds in the IDE easier. If I find the time I might try to extend this to other scenarios where a single configuration environment is desired. Help in this are is appreciated since my knowledge of conan and CMake is rather limited.

The code itself should be rather platform independent except for the class `WindowsConsoleListener` which would have to be replaced with one for the corresponding platform. Other than that some (but hopefully not that many) non-standard idioms specific to the MSVC compiler might be present.

You need a fully licensed copy of CPLEX. The trial version has limits on the number of variables and constraints per optimization model. And the models of the SparsityDrivenDetector are way to huge for that. The structure of the project however should allow to exchange the optimizer for a different open source one at some point.
