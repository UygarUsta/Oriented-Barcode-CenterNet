cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

set(CUDA_DIRECTORY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6")
set(CMAKE_PREFIX_PATH "C:/Users/John/Desktop/libtorch_2.3")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED PATHS "C:/Users/John/Desktop/opencv/build")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" ${OpenCV_LIBS} )
include_directories(${OpenCV_INCLUDE_DIRS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
