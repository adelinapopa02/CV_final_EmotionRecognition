#include "FaceDetector.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

FaceDetector::FaceDetector(const std::string& cascade_file) : cascade_path(cascade_file) {
    // Load the Haar cascade classifier
    if (!face_cascade.load(cascade_path)) {
        throw std::runtime_error("Error loading cascade file: " + cascade_path);
    }
    std::cout << "Face detector initialized successfully!" << std::endl;
}

// Main face detector method using Viola-Jones algorithm
std::vector <cv::Rect> FaceDetector::detectFaces(const cv::Mat& image) {
    std::vector<cv::Rect> faces;
    cv::Mat gray_image;

    // Convert image to grayscale (for Haar cascade)
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else {
        gray_image = image.clone();
    }

    // Enhance image contrast for better detection
    cv::equalizeHist(gray_image, gray_image);

    face_cascade.detectMultiScale(gray_image, faces, 1.1, 3, 0, cv::Size(30, 30));
    std::cout << "Detected " << faces.size() << " face(s)" << std::endl;
    return faces;
}

// Save individual faces as separate images (required by the CNN in input)
std::vector<std::string> FaceDetector::saveFaceRegions(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::string& base_filename) {
    
    std::vector<std::string> face_files;
    
    for (size_t i = 0; i < faces.size(); i++) {
        // Extract face region with some padding
        cv::Rect expanded_face = faces[i];
        int padding = 20;
        
        // Add padding while ensuring we don't go outside image bounds
        expanded_face.x = std::max(0, expanded_face.x - padding);
        expanded_face.y = std::max(0, expanded_face.y - padding);
        expanded_face.width = std::min(image.cols - expanded_face.x, expanded_face.width + 2 * padding);
        expanded_face.height = std::min(image.rows - expanded_face.y, expanded_face.height + 2 * padding);

        // Extract and save face region
        cv::Mat face_roi = image(expanded_face);
        std::string face_filename = base_filename + "_face_" + std::to_string(i) + ".jpg";
        
        if (cv::imwrite(face_filename, face_roi)) {
            face_files.push_back(face_filename);
            std::cout << "Saved face " << i << " to: " << face_filename << std::endl;
        } else {
            std::cerr << "Error saving face " << i << " to: " << face_filename << std::endl;
        }
    }
    
    return face_files;
}

// Draw bounding boxes around faces
cv::Mat FaceDetector::drawFaceBoxes(const cv::Mat& image, const std::vector<cv::Rect>& faces) {
    cv::Mat result = image.clone();
    
    for (size_t i = 0; i < faces.size(); i++) {
        // Draw rectangle around face
        cv::rectangle(result, faces[i], cv::Scalar(0, 255, 0), 2);
        
        // Add face number label
        std::string label = "Face " + std::to_string(i);
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point label_origin(faces[i].x, faces[i].y - 10);
        
        // Draw background rectangle for text
        cv::rectangle(result, 
                     cv::Point(label_origin.x, label_origin.y - label_size.height),
                     cv::Point(label_origin.x + label_size.width, label_origin.y + baseline),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        // Draw text
        cv::putText(result, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(0, 0, 0), 1);
    }
    
    return result;
}