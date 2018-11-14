from conans import ConanFile, CMake, tools

def get_version():
    git = tools.Git()
    try:
        return "%s" % (git.get_revision()[:7]) # get first 7 chars of git sha1
    except:
        return None

class PolarsConan(ConanFile):
    name = "Polars"
    version = get_version()
    url="https://github.com/felix-org/polars"
    license="MIT License"
    description = "A C++ TimeSeries library that aims to mimic pandas Series"
    settings = "cppstd", "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"
    exports_sources = "../*", "!dependencies/*", "!build"
    requires = "Armadillo/9.200.1@felix/stable", "Date/2.4.1@felix/stable"

    def build(self):
        cmake = CMake(self)
        cmake.definitions["WITH_TESTS"] = "OFF"
        cmake.definitions["WITH_SUBMODULE_DEPENDENCIES"] = "OFF"
        cmake.definitions["BUILD_WITH_CONAN"] = "ON"
        cmake.configure()
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include", src="src/cpp")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["polars_cpp"]
