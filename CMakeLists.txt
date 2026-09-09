cmake_minimum_required(VERSION 3.16)
project(OpenNeedle VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The standalone OpenNeedle build currently targets Linux.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# cq.cpp includes engine.cpp, sdot.cpp and frontend.cpp.
add_library(needle2_native SHARED needle2/csrc/cq.cpp)
target_compile_features(needle2_native PRIVATE cxx_std_17)
set_target_properties(needle2_native PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(needle2_native PRIVATE Threads::Threads OpenMP::OpenMP_CXX)

include(GNUInstallDirs)
install(TARGETS needle2_native LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES needle2/csrc/frontend.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/needle2)
