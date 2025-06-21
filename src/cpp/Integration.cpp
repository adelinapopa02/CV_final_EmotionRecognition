#include "Integration.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

Integration::Integration(
    const std::string& cascade_file,
    const std::string& python_script,
    const std::string& emotion_model
) : face_detector(cascade_file), python_script_path(python_script), model_path(emotion_model) {
    
    // Check if Python script exists (only needed for complete integration)
    if (!std::filesystem::exists(python_script_path)) {
        std::cerr << "Warning: Python script not found: " << python_script_path << std::endl;
        std::cerr << "Complete integration mode will not be available." << std::endl;
    }
    
    std::cout << "System Integrator initialized!" << std::endl;
}

bool Integration::runFaceDetectionOnly(const std::string& image_path) {
    try {
        // Load input image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image from " << image_path << std::endl;
            return false;
        }

        std::cout << "\n=== FACE DETECTION MODE ===" << std::endl;
        std::cout << "Input image: " << image_path << std::endl;
        std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

        // Detect faces
        std::vector<cv::Rect> faces = face_detector.detectFaces(image);

        if (faces.empty()) {
            std::cout << "No faces detected in the image." << std::endl;
            return true; // Not an error, just no faces found
        }

        // Generate output filenames
        std::string base_filename = getBaseFilename(image_path);
        std::string output_image = base_filename + "_faces_detected.jpg";

        // Draw bounding boxes and save result
        cv::Mat result_image = face_detector.drawFaceBoxes(image, faces);
        if (!cv::imwrite(output_image, result_image)) {
            std::cerr << "Error: Could not save result image to " << output_image << std::endl;
            return false;
        }
        std::cout << "Result saved to: " << output_image << std::endl;

        // Save individual face regions
        std::vector<std::string> face_files = face_detector.saveFaceRegions(image, faces, base_filename);

        std::cout << "\n=== FACE DETECTION SUMMARY ===" << std::endl;
        std::cout << "Total faces detected: " << faces.size() << std::endl;
        std::cout << "Face region files created: " << face_files.size() << std::endl;
        std::cout << "Face detection completed successfully!" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in face detection: " << e.what() << std::endl;
        return false;
    }
}

bool Integration::runCompleteIntegration(const std::string& image_path) {
    try {
        // Load input image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image from " << image_path << std::endl;
            return false;
        }

        std::cout << "\n=== COMPLETE INTEGRATION MODE ===" << std::endl;
        std::cout << "Input image: " << image_path << std::endl;
        std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

        // Step 1: Face Detection
        std::cout << "\nStep 1: Face Detection" << std::endl;
        std::vector<cv::Rect> faces = face_detector.detectFaces(image);

        if (faces.empty()) {
            std::cout << "No faces detected. Process completed." << std::endl;
            return true;
        }

        // Step 2: Save face regions
        std::cout << "\nStep 2: Extracting Face Regions" << std::endl;
        std::string base_filename = getBaseFilename(image_path);
        std::vector<std::string> face_files = face_detector.saveFaceRegions(image, faces, base_filename);

        // Step 3: Emotion Recognition
        std::cout << "\nStep 3: Emotion Recognition" << std::endl;
        std::vector<std::string> emotions = runEmotionRecognition(base_filename);

        // Step 4: Create final result
        std::cout << "\nStep 4: Creating Final Result" << std::endl;
        cv::Mat final_result = createFinalResult(image, faces, emotions);
        
        std::string final_output = base_filename + "_final_result.jpg";
        if (!cv::imwrite(final_output, final_result)) {
            std::cerr << "Error: Could not save final result to " << final_output << std::endl;
            return false;
        }
        std::cout << "Final result saved to: " << final_output << std::endl;

        // Step 5: Print summary
        std::cout << "\n=== INTEGRATION SUMMARY ===" << std::endl;
        std::cout << "Total faces detected: " << faces.size() << std::endl;
        std::cout << "Emotions recognized: " << emotions.size() << std::endl;
        
        for (size_t i = 0; i < std::min(faces.size(), emotions.size()); i++) {
            std::cout << "Face " << i << ": " << emotions[i] << std::endl;
        }
        
        std::cout << "\nOutput files created:" << std::endl;
        std::cout << "- " << final_output << " (final result with annotations)" << std::endl;
        std::cout << "- " << base_filename + "_emotions.txt (emotion predictions)" << std::endl;
        std::cout << "- " << base_filename + "_emotion_results.json (detailed results)" << std::endl;
        
        for (const auto& face_file : face_files) {
            std::cout << "- " << face_file << " (extracted face region)" << std::endl;
        }

        std::cout << "Complete integration completed successfully!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in complete integration: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> Integration::runEmotionRecognition(const std::string& base_filename) {
    std::vector<std::string> emotions;
    
    // Construct Python command
    auto quote = [](const std::string &s){ return "\"" + s + "\""; };
    std::string python_command = 
        "python3 " 
        + quote(python_script_path) + " " 
        + quote(base_filename)                       
        + (model_path.empty() ? "" : " " + quote(model_path));
    
    std::cout << "Running emotion recognition..." << std::endl;
    std::cout << "Command: " << python_command << std::endl;
    
    // Execute Python script
    int result = std::system(python_command.c_str());
    
    if (result != 0) {
        std::cerr << "Warning: Emotion recognition script returned non-zero exit code: " 
                  << result << std::endl;
    }
    
    // Read emotions from output file
    std::string emotions_file = base_filename + "_emotions.txt";
    std::ifstream file(emotions_file);
    
    if (file.is_open()) {
        std::string emotion;
        while (std::getline(file, emotion)) {
            if (!emotion.empty()) {
                emotions.push_back(emotion);
            }
        }
        file.close();
        std::cout << "Successfully read " << emotions.size() << " emotion predictions" << std::endl;
    } else {
        std::cerr << "Warning: Could not read emotions file: " << emotions_file << std::endl;
    }
    
    return emotions;
}

cv::Mat Integration::createFinalResult(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions) {
    
    cv::Mat result = image.clone();
    
    for (size_t i = 0; i < faces.size(); i++) {
        // Draw rectangle around face
        cv::rectangle(result, faces[i], cv::Scalar(0, 255, 0), 3);
        
        // Prepare label text
        std::string label = "Face " + std::to_string(i);
        if (i < emotions.size() && !emotions[i].empty()) {
            label += ": " + emotions[i];
        }
        
        // Calculate text size and position
        int baseline = 0;
        double font_scale = 0.7;
        int thickness = 2;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                            font_scale, thickness, &baseline);
        
        // Position label above the face rectangle
        cv::Point label_origin(faces[i].x, faces[i].y - 10);
        
        // Draw background rectangle for text
        cv::rectangle(result, 
                     cv::Point(label_origin.x, label_origin.y - label_size.height - 5),
                     cv::Point(label_origin.x + label_size.width + 10, label_origin.y + baseline),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        // Draw text
        cv::putText(result, label, cv::Point(label_origin.x + 5, label_origin.y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness);
    }
    
    return result;
}

std::string Integration::getBaseFilename(const std::string& filepath) {
    std::filesystem::path path(filepath);
    // Always save outputs to the output directory (created by CMake)
    return "../output/" + path.stem().string();
}