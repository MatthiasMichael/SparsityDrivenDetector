# SparsityDrivenDetector
This is code provided for the paper *Fusing Shape-from-Silhouette and the Sparsity Driven Detector for Camera-Based 3D Multi-Object Localization with Occlusions*.
Currently it is only tested on Windows with Visual Studio 2017. To make it usable on other platforms some conanfiles would have to be adapted as well as the class `WindowsConsoleListener`.

## Requirements
This project has several dependencies (OpenCV, OpenSceneGraph, ...) and uses [conan](https://conan.io) together with CMake to manage most of them.
In addition to the official `conan-center`repository you will also need *bincrafters* which can be added with:

    conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
    
At the time of writing however, OpenSceneGraph can not be found in these remotes. But there exists a corresponding recipe in https://github.com/bincrafters/conan-openscenegraph. The package can be installed with:

    git clone https://github.com/bincrafters/conan-openscenegraph.git
    cd conan-openscenegraph
    conan create . bincrafters/stable
    
## Non-Conan Dependencies
There are some libraries that cannot be found in the remotes and which need to be installed manually These are:
* Qt5
* CUDA 10
* CPLEX 12.7.1 (https://www.ibm.com/analytics/cplex-optimizer)

CPLEX is proprietary software but can be used freely for academic and research purposes. I wrote a small conanfile which simply extracts all headers and libraries from the CPLEX install folder and provides them as a conan package. This file can be found in my conan-CPLEX repository. It can be installed with:

    git clone https://github.com/MatthiasMichael/conan-CPLEX.git
    cd conan-CPLEX
    conan export . MatthiasMichael/stable
    conan install CPLEX/12.7.1@MatthiasMichael/stable --build missing
    
## Dependencies
Any other external dependecy should automatically managed by conan. This project however depends on several other libraries that are provided in my other repositories. These are:
* NamedType
* Parametrizable
* OsgVisualization
* Roi3DF
* Geometry

and have to be cloned individually. 

Then you can export each one as a conan package:

    git clone <xyz.git>
    cd <xyz>
    conan export . MatthiasMichael/stable
    conan install <xyz>/1.0@MatthiasMichael/stable -- build missing
    
Alternatively you can use the conan workspace file provieded in `conanws-SparsityDrivenDetector` to build them all at once.

## Building
If you followed the non-workspace approach you can simply install all dependencies and let conan build the project:

    git clone https://github.com/MatthiasMichael/SparsityDrivenDetector.git
    cd SparsityDrivenDetector
    mkdir vc141
    cd vc141
    conan install .. 
    conan build ..
    
The workspace approach requires all Libraries to be checked out in the same folder. If thats not the case, the paths in SparsityDrivenDetector.yml need to be adjusted accordingly.
Then in the `conanws-SparsityDrivenDetector`folder simply call:

    mkdir vc141
    cd vc141
    conan workspace install ../SparsityDrivenDetector.yml
    
Then invoke CMake in your preferred way.

## Running
TODO
