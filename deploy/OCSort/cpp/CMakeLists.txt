cmake_minimum_required(VERSION 3.10)
project(libocsort)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /O2") 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")

set(CMAKE_CXX_STANDARD 17)

# Linker external library
SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

# add_subdirectory(src)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# note：test with yolo inference
add_executable(test mutilthread.cpp ../detector/inference.h ../detector/inference.cpp)
target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})


