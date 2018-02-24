cmake_minimum_required(VERSION 3.5)
project(numpy)

include(common.cmake)

find_package(PythonLibs 3 REQUIRED)

execute_process(
  COMMAND "python3" -c "import numpy; print(numpy.get_include(), end='')"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  RESULT_VARIABLE NUMPY_NOTFOUND)

if (NUMPY_NOTFOUND)
  message(FATAL_ERROR "Numpy not found")
endif()

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

add_library(quantize SHARED quantize.cpp)

target_include_directories(quantize PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

set_target_properties(quantize PROPERTIES PREFIX "")
if (APPLE)
  set_target_properties(quantize PROPERTIES SUFFIX .so)
endif()
