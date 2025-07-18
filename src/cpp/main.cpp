#include "FaceDetector.h"
#include "Integration.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <thread>

void printUsage(const char* program_name) {
    std::cout << "Face Detection and Emotion Recognition System" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << program_name << " <mode> <input> [options...]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  --detect-only     Run face detection only" << std::endl;
    std::cout << "  --integrate       Run complete pipeline (face detection + emotion recognition)" << std::endl;
    std::cout << "  --batch-detect    Run face detection only on all images in directory" << std::endl;
    std::cout << "  --batch-integrate Run complete pipeline on all images in directory" << std::endl;
    std::cout << std::endl;
    std::cout << "Input:" << std::endl;
    std::cout << "  <input>           Path to image file (for single image modes)" << std::endl;
    std::cout << "                    Path to directory (for batch modes)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cascade <path>  Path to Haar cascade file" << std::endl;
    std::cout << "  --script <path>   Path to emotion recognition Python script" << std::endl;
    std::cout << "  --model <path>    Path to emotion recognition model (optional with DeepFace)" << std::endl;
    std::cout << "  --extensions <ext1,ext2,...>  Image extensions to process (default: jpg,jpeg,png,bmp)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Single image processing" << std::endl;
    std::cout << "  " << program_name << " --detect-only ../data/images/happy_1.jpg" << std::endl;
    std::cout << "  " << program_name << " --integrate ../data/images/happy_1.jpg" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Batch processing - all images in directory" << std::endl;
    std::cout << "  " << program_name << " --batch-integrate ../data/images/" << std::endl;
    std::cout << "  " << program_name << " --batch-detect ../data/images/" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Batch processing with custom extensions" << std::endl;
    std::cout << "  " << program_name << " --batch-integrate ../data/images/ --extensions jpg,png" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Custom paths for batch processing" << std::endl;
    std::cout << "  " << program_name << " --batch-integrate ../data/images/ \\" << std::endl;
    std::cout << "    --cascade ../models/haarcascade_frontalface_alt.xml \\" << std::endl;
    std::cout << "    --script ../src/python/emotion_recognition.py" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This version uses DeepFace for emotion recognition." << std::endl;
    std::cout << "      Models will be downloaded automatically on first use." << std::endl;
}

struct ProgramOptions {
    std::string mode;
    std::string input_path;
    std::string cascade_path = "../models/haarcascade_frontalface_alt.xml";
    std::string script_path = "../src/python/emotion_recognition.py";
    std::string model_path = "";  // Empty for DeepFace (not needed)
    std::string output_dir = "./output";
    std::vector<std::string> extensions = {"jpg", "jpeg", "png", "bmp"};
    bool valid = false;
    bool is_batch = false;
};

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> findImageFiles(const std::string& directory, const std::vector<std::string>& extensions) {
    std::vector<std::string> image_files;
    
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        std::cerr << "Error: Directory does not exist: " << directory << std::endl;
        return image_files;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();
            
            // Remove leading dot from extension
            if (!extension.empty() && extension[0] == '.') {
                extension = extension.substr(1);
            }
            
            // Convert extension to lowercase for comparison
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if extension is in our list
            for (const auto& ext : extensions) {
                std::string ext_lower = ext;
                std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
                if (extension == ext_lower) {
                    image_files.push_back(entry.path().string());
                    break;
                }
            }
        }
    }
    
    // Sort files for consistent processing order
    std::sort(image_files.begin(), image_files.end());
    
    return image_files;
}

ProgramOptions parseArguments(int argc, char* argv[]) {
    ProgramOptions options;
    
    if (argc < 3) {
        return options; // Invalid - need at least mode and input path
    }
    
    options.mode = argv[1];
    options.input_path = argv[2];
    
    // Validate mode and set batch flag
    if (options.mode == "--detect-only" || options.mode == "--integrate") {
        options.is_batch = false;
    } else if (options.mode == "--batch-detect" || options.mode == "--batch-integrate") {
        options.is_batch = true;
    } else {
        std::cerr << "Error: Invalid mode '" << options.mode << "'" << std::endl;
        std::cerr << "Valid modes are: --detect-only, --integrate, --batch-detect, --batch-integrate" << std::endl;
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
            options.model_path = value;  // Optional for DeepFace
        } else if (option == "--output") {
            options.output_dir = value;
        } else if (option == "--extensions") {
            options.extensions = split(value, ',');
        } else {
            std::cerr << "Error: Unknown option '" << option << "'" << std::endl;
            return options;
        }
    }
    
    options.valid = true;
    return options;
}

bool processSingleImage(Integration& integrator, const std::string& image_path, const std::string& mode, int current = 0, int total = 0) {
    if (total > 1) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "[" << current << "/" << total << "] Processing: " << std::filesystem::path(image_path).filename().string() << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    } else {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Processing: " << std::filesystem::path(image_path).filename().string() << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    bool success = false;
    
    if (mode == "--detect-only" || mode == "--batch-detect") {
        success = integrator.runFaceDetectionOnly(image_path);
    } else if (mode == "--integrate" || mode == "--batch-integrate") {
        success = integrator.runCompleteIntegration(image_path);
    }
    
    if (success) {
        std::cout << "✅ Successfully processed: " << std::filesystem::path(image_path).filename().string() << std::endl;
    } else {
        std::cout << "❌ Failed to process: " << std::filesystem::path(image_path).filename().string() << std::endl;
    }
    
    return success;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ProgramOptions options = parseArguments(argc, argv);
        
        if (!options.valid) {
            printUsage(argv[0]);
            return -1;
        }
        
        std::cout << "Face Detection and Emotion Recognition System (DeepFace)" << std::endl;
        std::cout << "=======================================================" << std::endl;
        std::cout << "Mode: " << options.mode << std::endl;
        std::cout << "Input: " << options.input_path << std::endl;
        std::cout << "Cascade: " << options.cascade_path << std::endl;
        
        if (options.mode == "--integrate" || options.mode == "--batch-integrate") {
            std::cout << "Script: " << options.script_path << std::endl;
            std::cout << "Emotion Recognition: DeepFace (models downloaded automatically)" << std::endl;
        }
        
        if (options.is_batch) {
            std::cout << "Extensions: ";
            for (size_t i = 0; i < options.extensions.size(); ++i) {
                std::cout << options.extensions[i];
                if (i < options.extensions.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        // Initialize system integrator
        Integration integrator(
            options.cascade_path,
            options.script_path,
            options.model_path  // Empty for DeepFace
        );
        
        std::vector<std::string> image_files;
        
        if (options.is_batch) {
            // Batch processing - find all image files in directory
            image_files = findImageFiles(options.input_path, options.extensions);
            
            if (image_files.empty()) {
                std::cerr << "No image files found in directory: " << options.input_path << std::endl;
                std::cerr << "Looking for extensions: ";
                for (size_t i = 0; i < options.extensions.size(); ++i) {
                    std::cerr << options.extensions[i];
                    if (i < options.extensions.size() - 1) std::cerr << ", ";
                }
                std::cerr << std::endl;
                return -1;
            }
            
            std::cout << "\nFound " << image_files.size() << " image file(s) to process" << std::endl;
            
            // Show list of files to be processed
            std::cout << "\nFiles to process:" << std::endl;
            for (const auto& file : image_files) {
                std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
            }
            
            if (options.mode == "--batch-integrate") {
                std::cout << "\nNote: DeepFace models will be downloaded on first use." << std::endl;
                std::cout << "      This may take a moment for the first image." << std::endl;
            }
            
        } else {
            // Single image processing
            image_files.push_back(options.input_path);
        }
        
        // Process images
        int successful_count = 0;
        int failed_count = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < image_files.size(); ++i) {
            bool success = processSingleImage(
                integrator, 
                image_files[i], 
                options.mode, 
                static_cast<int>(i + 1), 
                static_cast<int>(image_files.size())
            );
            
            if (success) {
                successful_count++;
            } else {
                failed_count++;
            }
            
            // Add small delay between processing to avoid overwhelming the system
            if (options.is_batch && i < image_files.size() - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        // Final summary
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PROCESSING SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total images processed: " << image_files.size() << std::endl;
        std::cout << "Successful: " << successful_count << std::endl;
        std::cout << "Failed: " << failed_count << std::endl;
        std::cout << "Success rate: " << (successful_count * 100.0 / image_files.size()) << "%" << std::endl;
        std::cout << "Total processing time: " << duration.count() << " seconds" << std::endl;
        std::cout << "Average time per image: " << (duration.count() / static_cast<double>(image_files.size())) << " seconds" << std::endl;
        
        if (failed_count > 0) {
            std::cout << "\nSome images failed to process. Check the error messages above." << std::endl;
        }
        
        if (successful_count > 0) {
            std::cout << "\n🎉 Processing completed!" << std::endl;
            std::cout << "Results saved to: " << options.output_dir << std::endl;
            
            if (options.is_batch) {
                std::cout << "\nTo run evaluation on all processed images:" << std::endl;
                std::cout << "  ./run_evaluation.sh" << std::endl;
            }
            
            return (failed_count == 0) ? 0 : 1;
        } else {
            std::cerr << "\n❌ All images failed to process!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}