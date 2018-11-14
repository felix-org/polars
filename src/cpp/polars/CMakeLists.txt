
# Set source directory
set(CPP_SOURCE_DIR "src/cpp/polars")

# Build sources list
set(
        CPP_SOURCES
        "${CPP_SOURCE_DIR}/numc.h"
        "${CPP_SOURCE_DIR}/numc.cpp"
        "${CPP_SOURCE_DIR}/Series.cpp"
        "${CPP_SOURCE_DIR}/Series.h"
        "${CPP_SOURCE_DIR}/TimeSeries.h"
        "${CPP_SOURCE_DIR}/TimeSeriesMask.h"
        "${CPP_SOURCE_DIR}/SeriesMask.cpp"
        "${CPP_SOURCE_DIR}/SeriesMask.h"
        "${CPP_SOURCE_DIR}/WindowProcessor.cpp"
        "${CPP_SOURCE_DIR}/WindowProcessor.h"
)

if(BUILD_WITH_CONAN)
  include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  conan_basic_setup(TARGETS)

  set(polars_dep_armadillo CONAN_PKG::Armadillo)
  set(polars_dep_date CONAN_PKG::Date)
else()
  set(polars_dep_armadillo armadillo)
  set(polars_dep_date date_interface)
endif()

add_library(polars_cpp ${CPP_SOURCES})
target_include_directories(polars_cpp PUBLIC ${Polars_SOURCE_DIR}/src/cpp)
target_link_libraries(polars_cpp ${polars_dep_armadillo} ${polars_dep_date})
