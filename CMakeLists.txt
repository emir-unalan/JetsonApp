cmake_minimum_required(VERSION 3.10)

project(jetsonApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TENSORRT_ROOT "/usr/local/tensorrt")

find_package(OpenCV 4.13 REQUIRED PATHS /usr/local)
find_package(CUDAToolkit REQUIRED)

link_directories(${TENSORRT_ROOT}/lib)

add_executable(jetsonApp main.cpp inference.cpp preprocess.cu postprocess.cu)

target_include_directories(jetsonApp PRIVATE  
    ${OpenCV_INCLUDE_DIRS} 
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_directories(jetsonApp PRIVATE ${TENSORRT_ROOT}/lib)
target_link_libraries(jetsonApp ${OpenCV_LIBS} nvinfer nvinfer_plugin nvonnxparser cudart)

set_target_properties(jetsonApp PROPERTIES
    BUILD_RPATH "${TENSORRT_ROOT}/lib"
    INSTALL_RPATH "${TENSORRT_ROOT}/lib"
)
