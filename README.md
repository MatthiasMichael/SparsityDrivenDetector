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
    conan install <package>/1.0@MatthiasMichael/stable --build missing
    
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
### Build only works with Visual Studio
Currenty the build process is tuned towards Windows and Visual Studio. Conan is explicitly invoked with the cmake-multi generator to make switching between Release and Debug builds in the IDE easier. If I find the time I might try to extend this to other scenarios where a single configuration environment is desired. Help in this are is appreciated since my knowledge of conan and CMake is rather limited.

### Cannot build in RelWithDebInfo configuration
Only Debug and Release configurations are working which is due to the cmake-multi generator of conan. It does not automatically install Release dependencies for the RelWithDebInfo configuration (which is what you might expect from the Windows environment). Instead it tries to get build all debencies with `build_type=RelWithDebInfo` which is just not supported by some of the packages and results in a failure.

### Some parts of the code are windows specific
The code itself should be rather platform independent except for the class `WindowsConsoleListener` which would have to be replaced with one for the corresponding platform. Other than that some (but hopefully not that many) non-standard idioms specific to the MSVC compiler might be present.

### Trial version of CPLEX is not sufficient
You need a fully licensed copy of CPLEX. The trial version has limits on the number of variables and constraints per optimization model. And the models of the SparsityDrivenDetector are way to huge for that. 
Apparently IBM currently has problems distributing the academic version of CPLEX. If that problem persists we will look into providing alternative optimizers (like Gurobi or other open source libraries)

### Build requires exactly CUDA 10 and Visual Studio 2017
We make use of several C++17 features that are not available in older versions of visual Visual Studio. However older CUDA versions do not support VS 2017 (CMake reports a missing toolchain). This also means that you currently cannot build this project for GPUs with Fermi architecture and older (like GeForce 400, 500, 600, GT-630) since they require SM20 which has been removed since CUDA 9.
These version mismatches are a mess we were not aware of during development since we thankfully had fairly new hardware. Maybe we'll look into adapting the code to make it buildable with VS 2015.

### CUDA compile options are set to -gencode arch=compute_50,code=sm_50
This is the lowest option allowed by CUDA 10 and should be supported by most modern GPUs. In theory it should be possible to instruct NVCC to compile for multiple target platforms at once. A problem with CMake generator expressions however prohibits that. If you need another target architecture you have to change line 62 of `shape_from_silhouette_cuda/cmake/CMakeLists.txt`
