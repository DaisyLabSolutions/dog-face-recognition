cmake_minimum_required(VERSION 3.2)
project(YOLOv5ncnn-vulkan)
SET(CMAKE_BUILD_TYPE Release)
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)  
find_package(Threads)

if(OPENMP_FOUND)
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()


# ncnn
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND ")
endif()

option(DAISYKIT_WITH_VULKAN "Build with Vulkan" ON)
if (DAISYKIT_WITH_VULKAN)
    find_package(Vulkan)
    # If not found Vulkan, disable GPU support
    if(NOT Vulkan_FOUND)
        message("Vulkan is not found. Disabling GPU support.")
        set(DAISYKIT_WITH_VULKAN OFF)
    endif()
    
    option(NCNN_VULKAN "" ${DAISYKIT_WITH_VULKAN})
endif()


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(main src/face_dog_detector.cpp
                    src/face_dog_extractor.cpp
                    src/ncnn_model.cpp 
                    src/image_model.cpp
                    src/tools.cpp 
                    src/main.cpp)
# add_executable(yolov5 src/yolov5.cpp)


target_link_libraries(
    main
    # yolov5 
    ncnn
    ${OpenCV_LIBS} 
    
)
