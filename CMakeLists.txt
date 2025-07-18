cmake_minimum_required(VERSION 3.10)
project(FaceEmotionDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Let CMake auto-detect architecture on macOS
if(APPLE)
    message(STATUS "Building for macOS - auto-detecting architecture")
endif()

# Find OpenCV package - try multiple approaches
# First try to find OpenCV in standard locations
find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    # Try Homebrew locations for different architectures
    if(APPLE)
        # Try Apple Silicon Homebrew location
        set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
        find_package(OpenCV QUIET PATHS "/opt/homebrew" NO_DEFAULT_PATH)
        
        # Try Intel Mac Homebrew location if Apple Silicon didn't work
        if(NOT OpenCV_FOUND)
            set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
            find_package(OpenCV QUIET PATHS "/usr/local" NO_DEFAULT_PATH)
        endif()
        
        # Try MacPorts location
        if(NOT OpenCV_FOUND)
            set(OpenCV_DIR "/opt/local/lib/cmake/opencv4")
            find_package(OpenCV QUIET PATHS "/opt/local" NO_DEFAULT_PATH)
        endif()
    endif()
endif()

# Final attempt - require OpenCV and let it fail with helpful message if not found
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found in standard locations.")
    message(STATUS "Please install OpenCV using one of these methods:")
    message(STATUS "  brew install opencv")
    message(STATUS "  sudo port install opencv4  (if using MacPorts)")
    message(STATUS "Or set OpenCV_DIR manually: cmake -DOpenCV_DIR=/path/to/opencv/cmake ..")
    find_package(OpenCV REQUIRED)
endif()

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
    COMMAND echo ""
    COMMAND echo "Setup Python environment first:"
    COMMAND echo "  python3 -m venv cv_native"
    COMMAND echo "  source cv_native/bin/activate"
    COMMAND echo "  pip install opencv-python deepface numpy matplotlib seaborn scikit-learn"
)