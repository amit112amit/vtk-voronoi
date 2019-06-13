cmake_minimum_required(VERSION 3.0.2)

project(OPS CXX)

cmake_policy(SET CMP0053 NEW)
cmake_policy(SET CMP0054 NEW)

set(CMAKE_CXX_STANDARD 14)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(dual DualMesh.cxx)
target_link_libraries(dual ${VTK_LIBRARIES} Eigen3::Eigen)
target_include_directories(dual PRIVATE ${VTK_INCLUDE_DIRS})
