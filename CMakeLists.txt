cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_FLAGS -g)

project(devices LANGUAGES CXX CUDA)

# message
message(STATUS "PROJECT_SOURCE_DIR is ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR is ${PROJECT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")

#output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

file(GLOB files "*/*.cu")
foreach (file ${files})
	message(STATUS "file is ${file}")
	string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
	message(STATUS "file name is ${exe}")
	add_executable(${exe} ${file})
	target_include_directories(${exe} PUBLIC /usr/local/cuda/samples/common/inc)
    target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda
        -G -src-in-ptx
        >)
    target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endforeach ()

