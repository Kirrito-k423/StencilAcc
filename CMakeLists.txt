cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CUDA_ARCHITECTURES 61)


project(devices LANGUAGES CXX CUDA)

# message
message(STATUS "PROJECT_SOURCE_DIR is ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR is ${PROJECT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# file(GLOB files "*/*.cu")
file(GLOB files "src/*.cu")
foreach (file ${files})
	# set(CMAKE_CUDA_FLAGS -g)
	message(STATUS "file is ${file}")
	string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
	message(STATUS "file name is ${exe}")
	add_executable(${exe} ${file})
	target_link_libraries(${exe} nvToolsExt)
	target_include_directories(${exe} PUBLIC /usr/local/cuda/samples/common/inc)
    target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
			-g
        	-G 
			-src-in-ptx
        >)
	target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
		--extended-lambda
		--expt-relaxed-constexpr
	>)
endforeach ()
# 4.2.3.2. --debug (-g)
# Generate debug information for host code.

# 4.2.3.3. --device-debug (-G)
# Generate debug information for device code.

# --source-in-ptx (-src-in-ptx)
# Interleave source in PTX.

# May only be used in conjunction with --device-debug or --generate-line-info.

file(GLOB files "src/*.cu")
foreach (file ${files})
	# set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O3 -v --use_fast_math") # old way: 原因是这个是全局的设置，会影响所有exe
	message(STATUS "file is ${file}")
	string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
	set(exe ${exe}_flags)
	message(STATUS "file name is ${exe}")
	add_executable(${exe} ${file})
	target_link_libraries(${exe} nvToolsExt)
	target_include_directories(${exe} PUBLIC /usr/local/cuda/samples/common/inc)
	target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        	-Xptxas 
			-O3 
			-v 
			--use_fast_math
	>)
    # target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    #     	-G 
	#		-src-in-ptx
    #     >)
    target_compile_options(${exe}  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
		--extended-lambda
		--expt-relaxed-constexpr
	>)
endforeach ()


# --extended-lambda (-extended-lambda)
# Allow __host__, __device__ annotations in lambda declarations.

# --expt-relaxed-constexpr (-expt-relaxed-constexpr)
# Experimental flag: Allow host code to invoke __device__constexpr functions, and device code to invoke __host__constexpr functions.

set(CMAKE_CXX_COMPILER g++)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "-pthread -fopenmp") # 把release的 -O3 -DNDEBUG覆盖掉

file(GLOB files "src/*.cpp")
foreach (file ${files})
	message(STATUS "file is ${file}")
	string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
	set(exe ${exe})
	message(STATUS "file name is ${exe}")
	add_executable(${exe} ${file})
	target_compile_options(${exe}  PRIVATE
		-pthread 
		-fopenmp
	>)
endforeach ()


file(GLOB files "src/*.cpp")
foreach (file ${files})
	message(STATUS "file is ${file}")
	string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
	set(exe ${exe}_flags)
	message(STATUS "file name is ${exe}")
	add_executable(${exe} ${file})
	target_compile_options(${exe}  PRIVATE
			-O3 
			-pthread 
			-fopenmp
	>)
endforeach ()