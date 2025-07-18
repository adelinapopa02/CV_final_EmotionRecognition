#!/bin/bash

# Face Detection & Emotion Recognition Environment Setup
# Fixed version for M4 Mac with proper package versions

# ANSI Color Codes
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}Face Detection & Emotion Recognition - Environment Setup${RESET}"
echo "======================================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory (where CMakeLists.txt is located)${RESET}"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "cv_venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${RESET}"
    rm -rf cv_venv
fi

# Create new virtual environment
echo -e "${CYAN}Creating Python virtual environment...${RESET}"
python3 -m venv cv_venv

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create virtual environment${RESET}"
    echo -e "${YELLOW}Make sure Python 3 is installed: python3 --version${RESET}"
    exit 1
fi

# Activate virtual environment
echo -e "${CYAN}Activating virtual environment...${RESET}"
source cv_venv/bin/activate

# Upgrade pip
echo -e "${CYAN}Upgrading pip...${RESET}"
pip install --upgrade pip

# Install packages in compatible order for M4 Mac
echo -e "${CYAN}Installing compatible packages (letting pip resolve versions)...${RESET}"

echo -e "${CYAN}  Installing NumPy (compatible version)...${RESET}"
pip install "numpy>=1.21.0,<2.0.0"

echo -e "${CYAN}  Installing OpenCV (compatible version)...${RESET}"
pip install "opencv-python>=4.5.0"

echo -e "${CYAN}  Installing TensorFlow (latest compatible)...${RESET}"
pip install tensorflow

echo -e "${CYAN}  Installing tf-keras (required for TensorFlow 2.19+)...${RESET}"
pip install tf-keras

echo -e "${CYAN}  Installing DeepFace (latest)...${RESET}"
pip install deepface

echo -e "${CYAN}  Installing additional packages...${RESET}"
pip install matplotlib seaborn scikit-learn

# Verify installation
echo -e "${CYAN}Verifying package installation...${RESET}"
python3 -c "
try:
    import numpy as np
    print('NumPy: ' + np.__version__)
    
    import cv2
    print('OpenCV: ' + cv2.__version__)
    
    import tensorflow as tf
    print('TensorFlow: ' + tf.__version__)
    
    from deepface import DeepFace
    print('DeepFace: imported successfully')
    
    import matplotlib
    print('Matplotlib: ' + matplotlib.__version__)
    
    import seaborn
    print('Seaborn: ' + seaborn.__version__)
    
    import sklearn
    print('Scikit-learn: ' + sklearn.__version__)
    
    print('')
    print('All packages installed successfully!')
    
except ImportError as e:
    print('Import error: ' + str(e))
    exit(1)
except Exception as e:
    print('Unexpected error: ' + str(e))
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Package verification failed!${RESET}"
    echo -e "${YELLOW}Trying alternative approach...${RESET}"
    
    # Try with explicit numpy version constraint
    echo -e "${CYAN}Reinstalling with explicit numpy<2.0 constraint...${RESET}"
    pip install --force-reinstall "numpy<2.0" opencv-python tensorflow tf-keras deepface matplotlib seaborn scikit-learn
    
    # Test again
    python3 -c "
try:
    import numpy as np
    import cv2
    import tensorflow as tf
    from deepface import DeepFace
    import matplotlib
    import seaborn
    import sklearn
    
    print('Alternative installation successful!')
    print('   NumPy: ' + np.__version__)
    print('   OpenCV: ' + cv2.__version__)
    print('   TensorFlow: ' + tf.__version__)
    print('   Matplotlib: ' + matplotlib.__version__)
    print('   DeepFace: Latest version')
    
except Exception as e:
    print('Still failing: ' + str(e))
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Package installation failed!${RESET}"
        echo -e "${YELLOW}Please see troubleshooting suggestions below.${RESET}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Setup completed successfully!${RESET}"
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo -e "${CYAN}1. Activate the virtual environment:${RESET}"
echo "   source cv_venv/bin/activate"
echo ""
echo -e "${CYAN}2. Build the C++ project:${RESET}"
echo "   mkdir -p build && cd build"
echo "   cmake .."
echo "   make"
echo ""
echo -e "${CYAN}3. Run face detection:${RESET}"
echo "   ./face_emotion_system --detect-only ../data/images/your_image.jpg"
echo ""
echo -e "${CYAN}4. Run complete pipeline:${RESET}"
echo "   ./face_emotion_system --integrate ../data/images/your_image.jpg"
echo ""
echo -e "${CYAN}5. Process multiple images:${RESET}"
echo "   ./face_emotion_system --batch-integrate ../data/images/"
echo ""
echo -e "${CYAN}6. Evaluate results:${RESET}"
echo "   python3 ../src/python/evaluation.py ../output ../data/labels"
echo ""
echo -e "${BOLD}Troubleshooting for M4 Mac:${RESET}"
echo -e "${YELLOW}If you still encounter issues:${RESET}"
echo "• Make sure you're using the system Python 3, not homebrew Python"
echo "• Try: python3 -m pip install --upgrade pip setuptools wheel"
echo "• Install Xcode command line tools: xcode-select --install"
echo "• If OpenCV fails: brew install opencv (then retry script)"
echo ""
echo -e "${BOLD}Important Notes:${RESET}"
echo -e "${YELLOW}• This setup uses modern package versions compatible with M4 Mac${RESET}"
echo -e "${YELLOW}• DeepFace will download models on first use (requires internet)${RESET}"
echo -e "${YELLOW}• TensorFlow will automatically use Apple's Metal GPU acceleration${RESET}"