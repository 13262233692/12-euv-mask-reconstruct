cmake_minimum_required(VERSION 3.18)
project(sem_preprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
find_package(pybind11 QUIET)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(sem_preprocess_core STATIC
    src/sem_preprocess.cpp
)
target_link_libraries(sem_preprocess_core ${OpenCV_LIBS})

if(pybind11_FOUND)
    pybind11_add_module(sem_preprocess src/sem_preprocess_pybind.cpp)
    target_link_libraries(sem_preprocess PRIVATE sem_preprocess_core)
endif()

enable_testing()
add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess sem_preprocess_core ${OpenCV_LIBS})
add_test(NAME test_preprocess COMMAND test_preprocess)
