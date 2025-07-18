#include "FaceDetector.h"
#include "Integration.h"
#include "Colors.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <thread>

void printUsage(const char* program_name) {
    std::cout << Colors::bold("Face Detection and Emotion Recognition System") << std::endl;
    std::cout << Colors::bold("=============================================") << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("DESCRIPTION:") << std::endl;
    std::cout << "  A computer vision system that detects faces using Viola-Jones algorithm" << std::endl;
    std::cout << "  and recognizes emotions using DeepFace deep learning models." << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("USAGE:") << std::endl;
    std::cout << "  " << program_name << " <mode> <input> [options...]" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("MODES:") << std::endl;
    std::cout << "  " << Colors::bold("--detect-only") << "     Face detection only (no emotion recognition)" << std::endl;
    std::cout << "  " << Colors::bold("--integrate") << "       Complete pipeline (face detection + emotion recognition)" << std::endl;
    std::cout << "  " << Colors::bold("--batch-detect") << "    Face detection on all images in directory" << std::endl;
    std::cout << "  " << Colors::bold("--batch-integrate") << " Complete pipeline on all images in directory" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("INPUT:") << std::endl;
    std::cout << "  <input>           Path to image file (for single image modes)" << std::endl;
    std::cout << "                    Path to directory (for batch modes)" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("OPTIONS:") << std::endl;
    std::cout << "  " << Colors::bold("--cascade") << " <path>  Path to Haar cascade file" << std::endl;
    std::cout << "                    (default: ../models/haarcascade_frontalface_alt.xml)" << std::endl;
    std::cout << "  " << Colors::bold("--script") << " <path>   Path to emotion recognition Python script" << std::endl;
    std::cout << "                    (default: ../src/python/emotion_recognition.py)" << std::endl;
    std::cout << "  " << Colors::bold("--extensions") << " <ext1,ext2,...>  Image extensions for batch processing" << std::endl;
    std::cout << "                    (default: jpg,jpeg,png,bmp)" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("EXAMPLES:") << std::endl;
    std::cout << "  # Single image processing" << std::endl;
    std::cout << "  " << program_name << " --detect-only ../data/images/happy_1.jpg" << std::endl;
    std::cout << "  " << program_name << " --integrate ../data/images/happy_1.jpg" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Batch processing" << std::endl;
    std::cout << "  " << program_name << " --batch-integrate ../data/images/" << std::endl;
    std::cout << "  " << program_name << " --batch-detect ../data/images/" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Custom parameters" << std::endl;
    std::cout << "  " << program_name << " --batch-integrate ../data/images/ \\" << std::endl;
    std::cout << "    --cascade ../models/haarcascade_frontalface_default.xml \\" << std::endl;
    std::cout << "    --extensions jpg,png" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("OUTPUT FILES:") << std::endl;
    std::cout << "  Results are saved to ../output/ directory:" << std::endl;
    std::cout << "  - [name]_faces_detected.jpg     Face detection visualization" << std::endl;
    std::cout << "  - [name]_final_result.jpg       Complete pipeline result (with emotions)" << std::endl;
    std::cout << "  - [name]_face_N.jpg             Individual face regions" << std::endl;
    std::cout << "  - [name]_unified_results.txt    Face coordinates and emotions (YOLO format)" << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("EVALUATION:") << std::endl;
    std::cout << "  To evaluate system performance against ground truth labels:" << std::endl;
    std::cout << "  " << Colors::bold("python3 ../src/python/evaluation.py ../output ../data/labels") << std::endl;
    std::cout << std::endl;
    std::cout << Colors::info("SYSTEM INFO:") << std::endl;
    std::cout << "  Face Detection: Viola-Jones algorithm (OpenCV Haar Cascades)" << std::endl;
    std::cout << "  Emotion Recognition: DeepFace with multiple CNN models" << std::endl;
    std::cout << "  Supported Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral" << std::endl;
    std::cout << "  Image Formats: JPEG, PNG, BMP" << std::endl;
}

struct ProgramOptions {
    std::string mode;
    std::string input_path;
    std::string cascade_path = "../models/haarcascade_frontalface_alt.xml";
    std::string script_path = "../src/python/emotion_recognition.py";
    std::string model_path = "";  // Empty for DeepFace (not needed)
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
        std::cerr << Colors::error("Error: Invalid mode '") << options.mode << Colors::error("'") << std::endl;
        std::cerr << Colors::warning("Valid modes: --detect-only, --integrate, --batch-detect, --batch-integrate") << std::endl;
        return options;
    }
    
    // Parse optional arguments
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << Colors::error("Error: Option '") << argv[i] << Colors::error("' requires a value") << std::endl;
            return options;
        }
        
        std::string option = argv[i];
        std::string value = argv[i + 1];
        
        if (option == "--cascade") {
            options.cascade_path = value;
        } else if (option == "--script") {
            options.script_path = value;
        } else if (option == "--extensions") {
            options.extensions = split(value, ',');
        } else {
            std::cerr << Colors::error("Error: Unknown option '") << option << Colors::error("'") << std::endl;
            std::cerr << Colors::warning("Valid options: --cascade, --script, --extensions") << std::endl;
            return options;
        }
    }
    
    options.valid = true;
    return options;
}

void printSystemInfo(const ProgramOptions& options) {
    std::cout << Colors::bold("Face Detection and Emotion Recognition System") << std::endl;
    std::cout << Colors::bold("=============================================") << std::endl;
    std::cout << Colors::info("Mode: ") << options.mode << std::endl;
    std::cout << Colors::info("Input: ") << options.input_path << std::endl;
    std::cout << Colors::info("Cascade: ") << options.cascade_path << std::endl;
    
    if (options.mode == "--integrate" || options.mode == "--batch-integrate") {
        std::cout << Colors::info("Python Script: ") << options.script_path << std::endl;
        std::cout << Colors::info("Emotion Models: ") << "DeepFace (auto-download)" << std::endl;
    }
    
    if (options.is_batch) {
        std::cout << Colors::info("Image Extensions: ");
        for (size_t i = 0; i < options.extensions.size(); ++i) {
            std::cout << options.extensions[i];
            if (i < options.extensions.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    std::cout << Colors::info("Output Directory: ") << "../output/" << std::endl;
    std::cout << std::endl;
}

bool processSingleImage(Integration& integrator, const std::string& image_path, const std::string& mode, int current = 0, int total = 0) {
    if (total > 1) {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "[" << current << "/" << total << "] Processing: " << std::filesystem::path(image_path).filename().string() << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    } else {
        std::cout << std::string(60, '=') << std::endl;
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
        std::cout << Colors::success("[SUCCESS] ") << "Successfully processed: " << std::filesystem::path(image_path).filename().string() << std::endl;
    } else {
        std::cout << Colors::error("[FAILED] ") << "Failed to process: " << std::filesystem::path(image_path).filename().string() << std::endl;
    }
    
    return success;
}

void printFinalSummary(int successful_count, int failed_count, int total_count, 
                      std::chrono::seconds duration, const std::string& mode) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << Colors::bold("PROCESSING SUMMARY") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << Colors::info("Total images processed: ") << total_count << std::endl;
    std::cout << Colors::success("Successful: ") << successful_count << std::endl;
    std::cout << Colors::error("Failed: ") << failed_count << std::endl;
    std::cout << Colors::info("Success rate: ") << (successful_count * 100.0 / total_count) << "%" << std::endl;
    std::cout << Colors::info("Total processing time: ") << duration.count() << " seconds" << std::endl;
    if (total_count > 0) {
        std::cout << Colors::info("Average time per image: ") << (duration.count() / static_cast<double>(total_count)) << " seconds" << std::endl;
    }
    
    if (failed_count > 0) {
        std::cout << "\n" << Colors::warning("[WARNING] ") << "Some images failed to process. Check error messages above." << std::endl;
    }
    
    if (successful_count > 0) {
        std::cout << "\n" << Colors::success("[COMPLETED] ") << "Processing completed successfully!" << std::endl;
        std::cout << Colors::info("Results saved to: ") << "../output/" << std::endl;
        
        std::cout << "\n" << Colors::bold("NEXT STEPS:") << std::endl;
        if (mode == "--batch-integrate" || mode == "--batch-detect") {
            std::cout << Colors::info("• Evaluate system performance:") << std::endl;
            std::cout << "  " << Colors::bold("python3 ../src/python/evaluation.py ../output ../data/labels") << std::endl;
            std::cout << Colors::info("• View individual results in ../output/ directory") << std::endl;
        } else {
            std::cout << Colors::info("• Check results in ../output/ directory") << std::endl;
            std::cout << Colors::info("• Run batch processing on full dataset:") << std::endl;
            std::cout << "  " << Colors::bold("./face_emotion_system --batch-integrate ../data/images/") << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ProgramOptions options = parseArguments(argc, argv);
        
        if (!options.valid) {
            printUsage(argv[0]);
            return -1;
        }
        
        // Print system information
        printSystemInfo(options);
        
        // Validate input paths
        if (!options.is_batch) {
            if (!std::filesystem::exists(options.input_path)) {
                std::cerr << "Error: Input file does not exist: " << options.input_path << std::endl;
                return -1;
            }
        }
        
        if (!std::filesystem::exists(options.cascade_path)) {
            std::cerr << Colors::error("Error: Cascade file does not exist: ") << options.cascade_path << std::endl;
            std::cerr << Colors::warning("Please ensure the Haar cascade file is available.") << std::endl;
            return -1;
        }
        
        // Initialize system integrator
        Integration integrator(
            options.cascade_path,
            options.script_path,
            options.model_path
        );
        
        std::vector<std::string> image_files;
        
        if (options.is_batch) {
            // Batch processing - find all image files in directory
            image_files = findImageFiles(options.input_path, options.extensions);
            
            if (image_files.empty()) {
                std::cerr << Colors::error("Error: No image files found in directory: ") << options.input_path << std::endl;
                std::cerr << Colors::info("Looking for extensions: ");
                for (size_t i = 0; i < options.extensions.size(); ++i) {
                    std::cerr << options.extensions[i];
                    if (i < options.extensions.size() - 1) std::cerr << ", ";
                }
                std::cerr << std::endl;
                return -1;
            }
            
            std::cout << Colors::success("Found ") << image_files.size() << Colors::success(" image file(s) to process") << std::endl;
            
            if (options.mode == "--batch-integrate") {
                std::cout << "\n" << Colors::warning("Note: ") << "DeepFace models will be downloaded on first use." << std::endl;
                std::cout << Colors::warning("This may take a moment for the first image.") << std::endl;
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
        
        // Print final summary
        printFinalSummary(successful_count, failed_count, static_cast<int>(image_files.size()), 
                         duration, options.mode);
        
        return (failed_count == 0) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\n" << Colors::error("[FATAL ERROR] ") << e.what() << std::endl;
        std::cerr << Colors::warning("Please check your input files and system configuration.") << std::endl;
        return -1;
    }
}