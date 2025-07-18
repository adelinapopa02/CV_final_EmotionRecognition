#include "Integration.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <map>

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

        // Generate output filenames
        std::string base_filename = getBaseFilename(image_path);

        if (faces.empty()) {
            std::cout << "No faces detected in the image." << std::endl;
            std::cout << "Creating empty results file." << std::endl;
            
            // Create empty unified results file
            std::vector<std::string> empty_emotions;
            saveUnifiedResults(image, faces, empty_emotions, base_filename);
            
            // Create empty detection result image (just the original image)
            std::string output_image = base_filename + "_faces_detected.jpg";
            if (!cv::imwrite(output_image, image)) {
                std::cerr << "Error: Could not save result image to " << output_image << std::endl;
                return false;
            }
            std::cout << "Result saved to: " << output_image << std::endl;
            
            std::cout << "\n=== FACE DETECTION SUMMARY ===" << std::endl;
            std::cout << "Total faces detected: 0" << std::endl;
            std::cout << "Face detection completed successfully!" << std::endl;
            
            return true;
        }

        // Save unified results (detection only, no emotions)
        std::vector<std::string> empty_emotions;
        saveUnifiedResults(image, faces, empty_emotions, base_filename);

        // Draw bounding boxes and save result
        std::string output_image = base_filename + "_faces_detected.jpg";
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

        // Generate output filenames
        std::string base_filename = getBaseFilename(image_path);

        if (faces.empty()) {
            std::cout << "No faces detected. Creating empty results file." << std::endl;
            
            // Create empty unified results file
            std::vector<std::string> empty_emotions;
            saveUnifiedResults(image, faces, empty_emotions, base_filename);
            
            // Create final result image (just the original image)
            std::string final_output = base_filename + "_final_result.jpg";
            if (!cv::imwrite(final_output, image)) {
                std::cerr << "Error: Could not save final result to " << final_output << std::endl;
                return false;
            }
            std::cout << "Final result saved to: " << final_output << std::endl;
            
            std::cout << "\n=== INTEGRATION SUMMARY ===" << std::endl;
            std::cout << "Total faces detected: 0" << std::endl;
            std::cout << "Emotions recognized: 0" << std::endl;
            std::cout << "\nOutput files created:" << std::endl;
            std::cout << "- " << final_output << " (final result)" << std::endl;
            std::cout << "- " << base_filename + "_unified_results.txt (empty results)" << std::endl;
            std::cout << "Complete integration completed successfully!" << std::endl;
            
            return true;
        }

        // Step 2: Save face regions
        std::cout << "\nStep 2: Extracting Face Regions" << std::endl;
        std::vector<std::string> face_files = face_detector.saveFaceRegions(image, faces, base_filename);

        // Step 3: Emotion Recognition
        std::cout << "\nStep 3: Emotion Recognition" << std::endl;
        std::vector<std::string> emotions = runEmotionRecognition(base_filename);

        // Step 4: Save unified results (emotion + coordinates)
        std::cout << "\nStep 4: Saving Unified Results" << std::endl;
        saveUnifiedResults(image, faces, emotions, base_filename);

        // Step 5: Create final result
        std::cout << "\nStep 5: Creating Final Result" << std::endl;
        cv::Mat final_result = createFinalResult(image, faces, emotions);
        
        std::string final_output = base_filename + "_final_result.jpg";
        if (!cv::imwrite(final_output, final_result)) {
            std::cerr << "Error: Could not save final result to " << final_output << std::endl;
            return false;
        }
        std::cout << "Final result saved to: " << final_output << std::endl;

        // Step 6: Clean up remaining temporary files
        std::cout << "\nStep 6: Cleaning up temporary files" << std::endl;
        
        // Delete emotion_results.json file (emotions.txt is already deleted)
        std::string json_file = base_filename + "_emotion_results.json";
        
        if (std::filesystem::exists(json_file)) {
            std::filesystem::remove(json_file);
            std::cout << "Deleted temporary file: " << json_file << std::endl;
        }
        
        // Step 7: Print summary
        std::cout << "\n=== INTEGRATION SUMMARY ===" << std::endl;
        std::cout << "Total faces detected: " << faces.size() << std::endl;
        std::cout << "Emotions recognized: " << emotions.size() << std::endl;
        
        for (size_t i = 0; i < std::min(faces.size(), emotions.size()); i++) {
            std::cout << "Face " << i << ": " << emotions[i] << std::endl;
        }
        
        std::cout << "\nOutput files created:" << std::endl;
        std::cout << "- " << final_output << " (final result with annotations)" << std::endl;
        std::cout << "- " << base_filename + "_unified_results.txt (unified emotion + coordinates)" << std::endl;
        
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
        "/Users/adelinapopa/miniforge3_arm64/envs/cv_native/bin/python3 " 
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
        
        // Delete the temporary emotions file immediately after reading
        std::filesystem::remove(emotions_file);
    } else {
        std::cerr << "Warning: Could not read emotions file: " << emotions_file << std::endl;
    }
    
    return emotions;
}

void Integration::saveUnifiedResults(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions, const std::string& base_filename) {
    std::string unified_file = base_filename + "_unified_results.txt";
    
    // Emotion to class ID mapping (matching your ground truth labels)
    std::map<std::string, int> emotion_to_class = {
        {"Angry", 0},
        {"Disgust", 1}, 
        {"Fear", 2},
        {"Happy", 3},
        {"Sad", 4},
        {"Surprise", 5},
        {"Neutral", 6}
    };
    
    std::ofstream file(unified_file);
    if (file.is_open()) {
        // If no faces detected, create empty file
        if (faces.empty()) {
            file.close();
            std::cout << "Empty unified results saved to: " << unified_file << std::endl;
            return;
        }
        
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            
            // Get emotion class ID
            int emotion_class = 6; // Default to Neutral if not found
            if (i < emotions.size()) {
                auto it = emotion_to_class.find(emotions[i]);
                if (it != emotion_to_class.end()) {
                    emotion_class = it->second;
                }
            }
            
            // Convert to YOLO format (normalized coordinates)
            double center_x = (face.x + face.width / 2.0) / image.cols;
            double center_y = (face.y + face.height / 2.0) / image.rows;
            double width = face.width / (double)image.cols;
            double height = face.height / (double)image.rows;
            
            // Write in format: emotion_class center_x center_y width height
            file << emotion_class << " " << center_x << " " << center_y << " " << width << " " << height << std::endl;
        }
        file.close();
        std::cout << "Unified results saved to: " << unified_file << std::endl;
    } else {
        std::cerr << "Error: Could not create unified results file: " << unified_file << std::endl;
    }
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