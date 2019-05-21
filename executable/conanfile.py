from conans import ConanFile, CMake
import sys
sys.path.insert(0, "..")
import dependencies as deps

class ExecutablesConan(ConanFile):
    name = "ExecutablesConan"
    version = "HEAD"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = deps.all

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir="%s/cmake" % self.source_folder)
        cmake.build()

    def imports(self):
        self.copy("*.dll", "", "bin")