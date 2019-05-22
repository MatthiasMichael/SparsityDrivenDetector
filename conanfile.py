from conans import ConanFile, CMake

class SparsityDrivenDetectorConan(ConanFile):
    name = "SparsityDrivenDetector"
    version = "HEAD"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = \
        "CPLEX/12.7.1@MatthiasMichael/stable", \
        "Geometry/1.0@MatthiasMichael/stable", \
        "OpenSSL/1.0.2r@conan/stable", \
        "OsgVisualization/1.0@MatthiasMichael/stable", \
        "Roi3DF/1.0@MatthiasMichael/stable", \
        "boost_filesystem/1.66.0@bincrafters/stable", \
        "boost_program_options/1.66.0@bincrafters/stable", \
        "ffmpeg/4.0.2@bincrafters/stable", \
        "libzip/1.5.1@bincrafters/stable", \
        "opencv/3.4.5@conan/stable", \
        "sqlite3/3.21.0@bincrafters/stable"

    def imports(self):
        self.copy("*.dll", "executable", "bin")

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir="%s/cmake" % self.source_folder)
        cmake.build()