cmake_minimum_required(VERSION 3.10)
project(FaceEmotionDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force ARM64 architecture on Apple Silicon Macs
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_SYSTEM_PROCESSOR "arm64")
    message(STATUS "Building for Apple Silicon (ARM64)")
endif()

# Find OpenCV package with specific paths for conda environment
set(OpenCV_DIR "/Users/adelinapopa/miniforge3_arm64/envs/cv_native/lib/cmake/opencv4")
find_package(OpenCV REQUIRED PATHS "/Users/adelinapopa/miniforge3_arm64/envs/cv_native" NO_DEFAULT_PATH)

# Print OpenCV information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add include directory so we can find the header files
include_directories(src/cpp/include)

# Create output directory at build time
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# Set compiler flags for Apple Silicon
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")
endif()

# Create shared library for face detection
add_library(face_detection_lib src/cpp/FaceDetector.cpp)
target_link_libraries(face_detection_lib ${OpenCV_LIBS})
if(APPLE)
    set_target_properties(face_detection_lib PROPERTIES OSX_ARCHITECTURES "arm64")
endif()

# Create shared library for integration
add_library(integration_lib src/cpp/Integration.cpp)
target_link_libraries(integration_lib face_detection_lib ${OpenCV_LIBS})
if(APPLE)
    set_target_properties(integration_lib PROPERTIES OSX_ARCHITECTURES "arm64")
endif()

# Main executable that can run both modes
add_executable(face_emotion_system src/cpp/main.cpp)
target_link_libraries(face_emotion_system integration_lib face_detection_lib ${OpenCV_LIBS})
if(APPLE)
    set_target_properties(face_emotion_system PROPERTIES OSX_ARCHITECTURES "arm64")
endif()

# Custom target to show usage
add_custom_target(usage
    COMMAND echo "Usage:"
    COMMAND echo "  ./face_emotion_system --detect-only <image_path>"
    COMMAND echo "  ./face_emotion_system --integrate <image_path>"
    COMMAND echo ""
    COMMAND echo "Examples:"
    COMMAND echo "  ./face_emotion_system --detect-only ../data/images/happy_1.jpg"
    COMMAND echo "  ./face_emotion_system --integrate ../data/images/happy_1.jpg"
)