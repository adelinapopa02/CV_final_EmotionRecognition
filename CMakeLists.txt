cmake_minimum_required(VERSION 3.10)
project(FaceEmotionDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add include directory so we can find the header files
include_directories(src/cpp/include)

# Create output directory at build time
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# Create shared library for face detection
add_library(face_detection_lib src/cpp/FaceDetector.cpp)
target_link_libraries(face_detection_lib ${OpenCV_LIBS})

# Create shared library for integration
add_library(integration_lib src/cpp/Integration.cpp)
target_link_libraries(integration_lib face_detection_lib ${OpenCV_LIBS})

# Main executable that can run both modes
add_executable(face_emotion_system src/cpp/main.cpp)
target_link_libraries(face_emotion_system integration_lib face_detection_lib ${OpenCV_LIBS})

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
