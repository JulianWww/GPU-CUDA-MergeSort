# CMakeList.txt : CMake project for sorter, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)
 
add_library(sorting STATIC
  sorter.cu
  sorter.h
  )
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(sorting PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( sorting
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(sorter test.cu)
 
set_property(TARGET sorter 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sorter PRIVATE particles)
 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET sorter 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
