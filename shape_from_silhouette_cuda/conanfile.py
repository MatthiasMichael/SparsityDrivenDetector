from conans import ConanFile, CMake
import sys
sys.path.insert(0, "..")
import dependencies as deps

class ShapeFromSilhouetteCudaConan(ConanFile):
    name = "ShapeFromSilhouetteCuda"
    version = "HEAD"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = deps.get(deps.environment, deps.shape_from_silhouette_common, deps.shape_from_silhouette)

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir="%s/cmake" % self.source_folder)
        cmake.build()