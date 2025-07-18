#include "FaceDetector.h"
#include "Colors.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

FaceDetector::FaceDetector(const std::string& cascade_file) : cascade_path(cascade_file) {
    // Load the Haar cascade classifier
    if (!face_cascade.load(cascade_path)) {
        throw std::runtime_error("Error loading cascade file: " + cascade_path);
    }
    std::cout << Colors::success("Face detector initialized successfully!") << std::endl;
}

// Multiple preprocessing strategies for debugging
cv::Mat FaceDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Convert to grayscale
    if (image.channels() == 3) {
        cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = image.clone();
    }
    
    // Just return simple histogram equalization for now
    cv::Mat result;
    cv::equalizeHist(processed, result);
    
    return result;
}

// Try multiple detection strategies to debug what works
std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& image) {
    cv::Mat gray_image = preprocessImage(image);
    
    std::cout << Colors::bold("=== DEBUGGING FACE DETECTION ===") << std::endl;
    std::cout << Colors::info("Image size: ") << image.cols << "x" << image.rows << std::endl;
    
    // Strategy 1: Very sensitive detection
    std::vector<cv::Rect> faces1;
    face_cascade.detectMultiScale(gray_image, faces1, 1.05, 2, 0, cv::Size(20, 20));
    std::cout << Colors::info("Strategy 1 (very sensitive): ") << faces1.size() << " faces" << std::endl;
    
    // Strategy 2: Standard detection
    std::vector<cv::Rect> faces2;
    face_cascade.detectMultiScale(gray_image, faces2, 1.1, 3, 0, cv::Size(30, 30));
    std::cout << Colors::info("Strategy 2 (standard): ") << faces2.size() << " faces" << std::endl;
    
    // Strategy 3: Conservative detection
    std::vector<cv::Rect> faces3;
    face_cascade.detectMultiScale(gray_image, faces3, 1.1, 5, 0, cv::Size(40, 40));
    std::cout << Colors::info("Strategy 3 (conservative): ") << faces3.size() << " faces" << std::endl;
    
    // Strategy 4: Try with different scale factors
    std::vector<cv::Rect> faces4;
    face_cascade.detectMultiScale(gray_image, faces4, 1.03, 3, 0, cv::Size(25, 25));
    std::cout << Colors::info("Strategy 4 (small scale factor): ") << faces4.size() << " faces" << std::endl;
    
    // Strategy 5: Try with original image (no preprocessing)
    cv::Mat original_gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, original_gray, cv::COLOR_BGR2GRAY);
    } else {
        original_gray = image.clone();
    }
    
    std::vector<cv::Rect> faces5;
    face_cascade.detectMultiScale(original_gray, faces5, 1.1, 3, 0, cv::Size(30, 30));
    std::cout << Colors::info("Strategy 5 (no preprocessing): ") << faces5.size() << " faces" << std::endl;
    
    // Choose the best strategy
    std::vector<cv::Rect> best_faces;
    std::string best_strategy = "None";
    
    // Priority: find the strategy that gives reasonable number of faces (1-5)
    if (faces2.size() >= 1 && faces2.size() <= 5) {
        best_faces = faces2;
        best_strategy = "Strategy 2 (standard)";
    } else if (faces5.size() >= 1 && faces5.size() <= 5) {
        best_faces = faces5;
        best_strategy = "Strategy 5 (no preprocessing)";
    } else if (faces3.size() >= 1 && faces3.size() <= 5) {
        best_faces = faces3;
        best_strategy = "Strategy 3 (conservative)";
    } else if (faces4.size() >= 1 && faces4.size() <= 5) {
        best_faces = faces4;
        best_strategy = "Strategy 4 (small scale)";
    } else if (faces1.size() >= 1 && faces1.size() <= 10) {
        best_faces = faces1;
        best_strategy = "Strategy 1 (very sensitive)";
    } else {
        // If nothing good, just take the one with most faces (up to 10)
        std::vector<std::pair<int, std::vector<cv::Rect>>> all_results = {
            {1, faces1}, {2, faces2}, {3, faces3}, {4, faces4}, {5, faces5}
        };
        
        for (auto& result : all_results) {
            if (result.second.size() > best_faces.size() && result.second.size() <= 10) {
                best_faces = result.second;
                best_strategy = "Strategy " + std::to_string(result.first) + " (fallback)";
            }
        }
    }
    
    std::cout << Colors::success("Selected: ") << best_strategy << Colors::success(" with ") << best_faces.size() << Colors::success(" faces") << std::endl;
    
    // Apply our filtering to the best result
    if (!best_faces.empty()) {
        best_faces = filterFalsePositives(image, best_faces);
        std::cout << Colors::info("After filtering: ") << best_faces.size() << " faces" << std::endl;
        
        best_faces = applyNMS(best_faces, 0.3);
        std::cout << Colors::info("After NMS: ") << best_faces.size() << " faces" << std::endl;
    }
    
    std::cout << Colors::bold("=== END DEBUGGING ===") << std::endl;
    
    return best_faces;
}

// Simple but effective filtering based on size and aspect ratio
std::vector<cv::Rect> FaceDetector::filterFalsePositives(const cv::Mat& image, const std::vector<cv::Rect>& faces) {
    std::vector<cv::Rect> filtered_faces;
    
    if (faces.empty()) return filtered_faces;
    
    // Calculate image statistics for size filtering - more reasonable for portraits
    int image_area = image.rows * image.cols;
    int min_face_area = image_area / 10000;  // At least 0.01% of image  
    int max_face_area = image_area / 3;      // At most 33% of image (increased for portraits)
    
    // Calculate average face size to help identify outliers
    int total_area = 0;
    for (const auto& face : faces) {
        total_area += face.area();
    }
    int avg_face_area = total_area / faces.size();
    
    std::cout << Colors::info("Filtering ") << faces.size() << Colors::info(" initial detections...") << std::endl;
    std::cout << Colors::info("Image area: ") << image_area 
              << Colors::info(", Min face area: ") << min_face_area 
              << Colors::info(", Max face area: ") << max_face_area 
              << Colors::info(", Avg face area: ") << avg_face_area << std::endl;
    
    for (size_t i = 0; i < faces.size(); i++) {
        const auto& face = faces[i];
        bool keep_face = true;
        std::string reject_reason = "";
        
        // 1. Size filtering
        int face_area = face.area();
        if (face_area < min_face_area) {
            keep_face = false;
            reject_reason = "too small (" + std::to_string(face_area) + " < " + std::to_string(min_face_area) + ")";
        } else if (face_area > max_face_area) {
            keep_face = false;
            reject_reason = "too large (" + std::to_string(face_area) + " > " + std::to_string(max_face_area) + ")";
        }
        
        // 2. Aspect ratio filtering - more strict
        if (keep_face) {
            double aspect_ratio = static_cast<double>(face.width) / face.height;
            if (aspect_ratio < 0.6 || aspect_ratio > 1.6) {  // More strict range
                keep_face = false;
                reject_reason = "bad aspect ratio (" + std::to_string(aspect_ratio) + ")";
            }
        }
        
        // 3. Minimum absolute size filtering - reasonable for portraits
        if (keep_face) {
            if (face.width < 30 || face.height < 30) {  // Reduced back to 30
                keep_face = false;
                reject_reason = "too small in pixels (" + std::to_string(face.width) + "x" + std::to_string(face.height) + ")";
            }
        }
        
        // 4. Keep only faces that are reasonably sized compared to the largest faces
        if (keep_face && faces.size() > 3) {  // Only filter when 4+ faces (allow more in portraits)
            // Find the largest face
            int max_area = 0;
            for (const auto& other_face : faces) {
                max_area = std::max(max_area, other_face.area());
            }
            
            // Reject faces that are much smaller than the largest (likely false positives)
            if (face_area < max_area / 20) {  // Less than 1/20th the size (more lenient)
                keep_face = false;
                reject_reason = "much smaller than largest face (" + std::to_string(face_area) + " vs " + std::to_string(max_area) + ")";
            }
        }
        
        if (keep_face) {
            filtered_faces.push_back(face);
            std::cout << Colors::success("✓ Kept face ") << i << Colors::success(": ") << face.width << "x" << face.height 
                      << Colors::success(" at (") << face.x << "," << face.y << Colors::success(")") << std::endl;
        } else {
            std::cout << Colors::error("✗ Rejected face ") << i << Colors::error(": ") << reject_reason << std::endl;
        }
    }
    
    return filtered_faces;
}

// Apply Non-Maximum Suppression to remove overlapping detections
std::vector<cv::Rect> FaceDetector::applyNMS(const std::vector<cv::Rect>& faces, double overlap_threshold) {
    if (faces.empty()) return faces;
    
    std::vector<cv::Rect> filtered_faces;
    std::vector<bool> suppressed(faces.size(), false);
    
    // Sort faces by area (larger faces first)
    std::vector<std::pair<int, cv::Rect>> indexed_faces;
    for (size_t i = 0; i < faces.size(); i++) {
        indexed_faces.push_back({static_cast<int>(i), faces[i]});
    }
    
    std::sort(indexed_faces.begin(), indexed_faces.end(), 
              [](const std::pair<int, cv::Rect>& a, const std::pair<int, cv::Rect>& b) {
                  return (a.second.width * a.second.height) > (b.second.width * b.second.height);
              });
    
    for (size_t i = 0; i < indexed_faces.size(); i++) {
        int idx_i = indexed_faces[i].first;
        if (suppressed[idx_i]) continue;
        
        cv::Rect rect_i = indexed_faces[i].second;
        filtered_faces.push_back(rect_i);
        
        // Suppress overlapping rectangles
        for (size_t j = i + 1; j < indexed_faces.size(); j++) {
            int idx_j = indexed_faces[j].first;
            if (suppressed[idx_j]) continue;
            
            cv::Rect rect_j = indexed_faces[j].second;
            
            // Calculate overlap ratio
            cv::Rect intersection = rect_i & rect_j;
            double overlap = static_cast<double>(intersection.area()) / 
                           std::min(rect_i.area(), rect_j.area());
            
            if (overlap > overlap_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    return filtered_faces;
}

// Save individual faces as separate images
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
            std::cout << Colors::success("Saved face ") << i << Colors::success(" to: ") << face_filename << std::endl;
        } else {
            std::cerr << Colors::error("Error saving face ") << i << Colors::error(" to: ") << face_filename << std::endl;
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