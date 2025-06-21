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

# Create models directory in build folder
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/models)

# Download Haar cascade file
set(HAAR_CASCADE_URL "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml")
set(HAAR_CASCADE_FILE "${CMAKE_BINARY_DIR}/models/haarcascade_frontalface_alt.xml")

if(NOT EXISTS ${HAAR_CASCADE_FILE})
    message(STATUS "Downloading Haar cascade classifier...")
    file(DOWNLOAD ${HAAR_CASCADE_URL} ${HAAR_CASCADE_FILE}
         SHOW_PROGRESS
         STATUS HAAR_DOWNLOAD_STATUS)
    
    list(GET HAAR_DOWNLOAD_STATUS 0 HAAR_STATUS_CODE)
    if(NOT HAAR_STATUS_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download Haar cascade file")
    endif()
    message(STATUS "Haar cascade downloaded successfully")
endif()

# Download emotion model file - manual download required
message(STATUS "Note: Emotion model requires manual download")
message(STATUS "Please download a FER2013 emotion model (.keras or .h5) to:")
message(STATUS "  ${CMAKE_BINARY_DIR}/models/emotion_model.keras")
message(STATUS "Suggested sources:")
message(STATUS "  - https://www.kaggle.com/datasets/msambare/fer2013")
message(STATUS "  - https://github.com/atulapra/Emotion-detection")

# Create a placeholder file to indicate manual download needed
file(WRITE ${CMAKE_BINARY_DIR}/models/DOWNLOAD_EMOTION_MODEL.txt
     "Please manually download an emotion recognition model to this directory.\n"
     "Rename it to 'emotion_model.keras' or 'emotion_model.h5'\n"
     "\n"
     "Suggested sources:\n"
     "- https://www.kaggle.com/datasets/msambare/fer2013\n"
     "- https://github.com/atulapra/Emotion-detection\n"
)

# Create shared library for face detection
add_library(face_detection_lib src/cpp/FaceDetector.cpp)
target_link_libraries(face_detection_lib ${OpenCV_LIBS})

# Create shared library for integration
add_library(integration_lib src/cpp/Integration.cpp)
target_link_libraries(integration_lib face_detection_lib ${OpenCV_LIBS})

# Main executable that can run both modes
add_executable(face_emotion_system src/cpp/main.cpp)
target_link_libraries(face_emotion_system integration_lib face_detection_lib ${OpenCV_LIBS})

# Custom target to manually download models if needed
add_custom_target(download_models
    COMMAND ${CMAKE_COMMAND} -E echo "Checking for required model files..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/models
    
    # Download Haar cascade if missing
    COMMAND test -f ${CMAKE_BINARY_DIR}/models/haarcascade_frontalface_alt.xml || 
            (echo "Downloading Haar cascade..." && 
             curl -L -o ${CMAKE_BINARY_DIR}/models/haarcascade_frontalface_alt.xml 
             ${HAAR_CASCADE_URL})
    
    # Download emotion model if missing
    COMMAND test -f ${CMAKE_BINARY_DIR}/models/emotion_model.keras || 
            test -f ${CMAKE_BINARY_DIR}/models/emotion_model.h5 ||
            (echo "Downloading emotion model..." && 
             curl -L -o ${CMAKE_BINARY_DIR}/models/emotion_model.keras 
             "https://github.com/petercunha/Emotion/raw/master/model/emotion_model.h5")
    
    COMMAND ${CMAKE_COMMAND} -E echo "Model download complete!"
    COMMENT "Downloading required model files"
)

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
    COMMAND echo "Models location: build/models/"
    COMMAND echo "  - haarcascade_frontalface_alt.xml"
    COMMAND echo "  - emotion_model.keras (or emotion_model.h5)"
    COMMAND echo ""
    COMMAND echo "To manually download models: make download_models"
)