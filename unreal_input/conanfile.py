from conans import ConanFile, CMake
import sys
sys.path.insert(0, "..")
import dependencies as deps

class UnrealInputConan(ConanFile):
    name = "UnrealInput"
    version = "HEAD"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = deps.all

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir="%s/cmake" % self.source_folder)
        cmake.build()