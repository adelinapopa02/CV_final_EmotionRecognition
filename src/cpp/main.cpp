#include "FaceDetector.h"
#include "Integration.h"
#include <iostream>
#include <string>

void printUsage(const char* program_name) {
    std::cout << "Face Detection and Emotion Recognition System" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << program_name << " <mode> <image_path> [options...]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  --detect-only     Run face detection only" << std::endl;
    std::cout << "  --integrate       Run complete pipeline (face detection + emotion recognition)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cascade <path>  Path to Haar cascade file" << std::endl;
    std::cout << "  --script <path>   Path to emotion recognition Python script" << std::endl;
    std::cout << "  --model <path>    Path to emotion recognition model" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Face detection only" << std::endl;
    std::cout << "  " << program_name << " --detect-only ../data/images/happy_1.jpg" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Complete pipeline" << std::endl;
    std::cout << "  " << program_name << " --integrate ../data/images/group_photo.jpg" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Custom cascade file" << std::endl;
    std::cout << "  " << program_name << " --detect-only image.jpg --cascade custom_cascade.xml" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Custom paths for complete integration" << std::endl;
    std::cout << "  " << program_name << " --integrate image.jpg \\" << std::endl;
    std::cout << "    --cascade models/haarcascade_frontalface_alt.xml \\" << std::endl;
    std::cout << "    --script ../src/python/emotion_recognition.py \\" << std::endl;
    std::cout << "    --model models/emotion_model.keras" << std::endl;
}

struct ProgramOptions {
    std::string mode;
    std::string image_path;
    std::string cascade_path = "models/haarcascade_frontalface_alt.xml";
    std::string script_path = "../src/python/emotion_recognition.py";
    std::string model_path = "models/emotion_model.keras";
    std::string output_dir   = "./output";
    bool valid = false;
};

ProgramOptions parseArguments(int argc, char* argv[]) {
    ProgramOptions options;
    
    if (argc < 3) {
        return options; // Invalid - need at least mode and image path
    }
    
    options.mode = argv[1];
    options.image_path = argv[2];
    
    // Validate mode
    if (options.mode != "--detect-only" && options.mode != "--integrate") {
        std::cerr << "Error: Invalid mode '" << options.mode << "'" << std::endl;
        std::cerr << "Valid modes are: --detect-only, --integrate" << std::endl;
        return options;
    }
    
    // Parse optional arguments
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Error: Option '" << argv[i] << "' requires a value" << std::endl;
            return options;
        }
        
        std::string option = argv[i];
        std::string value = argv[i + 1];
        
        if (option == "--cascade") {
            options.cascade_path = value;
        } else if (option == "--script") {
            options.script_path = value;
        } else if (option == "--model") {
            options.model_path = value;
        } else if (option == "--output") {
            options.output_dir = value;
        } else {
            std::cerr << "Error: Unknown option '" << option << "'" << std::endl;
            return options;
        }
    }
    
    options.valid = true;
    return options;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ProgramOptions options = parseArguments(argc, argv);
        
        if (!options.valid) {
            printUsage(argv[0]);
            return -1;
        }
        
        std::cout << "Face Detection and Emotion Recognition System" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Mode: " << options.mode << std::endl;
        std::cout << "Image: " << options.image_path << std::endl;
        std::cout << "Cascade: " << options.cascade_path << std::endl;
        
        if (options.mode == "--integrate") {
            std::cout << "Script: " << options.script_path << std::endl;
            std::cout << "Model: " << options.model_path << std::endl;
        }
        
        // Initialize system integrator
        Integration integrator(
            options.cascade_path,
            options.script_path,
            options.model_path
        );
        
        bool success = false;
        
        // Run the appropriate mode
        if (options.mode == "--detect-only") {
            success = integrator.runFaceDetectionOnly(options.image_path);
        } else if (options.mode == "--integrate") {
            success = integrator.runCompleteIntegration(options.image_path);
        }
        
        if (success) {
            std::cout << "\n🎉 Processing completed successfully!" << std::endl;
            return 0;
        } else {
            std::cerr << "\n❌ Processing failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}