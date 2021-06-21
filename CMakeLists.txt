cmake_minimum_required(VERSION 3.4...3.18)
project(ort_inference)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party/pybind11)
pybind11_add_module(ort_inference src/main.cpp)

if(APPLE)
    target_include_directories(
        ort_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-osx-x64-1.8.0/lib
    )
    find_library(
        ONNX_RUNTIME_LIB
        NAMES libonnxruntime.dylib # what to look for
        HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-osx-x64-1.8.0/lib # where to look
        NO_DEFAULT_PATH # do not search system default paths
    )
    target_compile_definitions(ort_inference PRIVATE OSX)
endif()

if(UNIX AND NOT APPLE)
    target_include_directories(
        ort_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.0/lib
    )
    find_library(
        ONNX_RUNTIME_LIB
        NAMES libonnxruntime.so.1.8.0 # what to look for
        HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.0/lib # where to look
        NO_DEFAULT_PATH # do not search system default paths
    )
    target_compile_definitions(ort_inference PRIVATE LINUX)
endif()

target_link_libraries(ort_inference PRIVATE ${ONNX_RUNTIME_LIB})

set(INFERENCE_HEADER_FILES
    src/inference.hpp
)
set(INFERENCE_CPP_FILES
    src/inference.cpp
)
add_library(INFERENCE_LIBRARY SHARED
    ${INFERENCE_HEADER_FILES}
    ${INFERENCE_CPP_FILES}
)
target_link_libraries(INFERENCE_LIBRARY PRIVATE ${ONNX_RUNTIME_LIB})
target_link_libraries(ort_inference PRIVATE INFERENCE_LIBRARY)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ort_inference PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
