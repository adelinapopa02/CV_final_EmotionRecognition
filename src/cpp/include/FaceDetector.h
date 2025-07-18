#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceDetector {
public:
    FaceDetector(const std::string& cascade_file);
    
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    std::vector<std::string> saveFaceRegions(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::string& base_filename);
    cv::Mat drawFaceBoxes(const cv::Mat& image, const std::vector<cv::Rect>& faces);

private:
    cv::CascadeClassifier face_cascade;
    std::string cascade_path;
    
    // Helper function for preprocessing
    cv::Mat preprocessImage(const cv::Mat& image);
    
    // Smart filtering to remove obvious false positives
    std::vector<cv::Rect> filterFalsePositives(const cv::Mat& image, const std::vector<cv::Rect>& faces);
    
    // Non-Maximum Suppression to remove overlapping detections
    std::vector<cv::Rect> applyNMS(const std::vector<cv::Rect>& faces, double overlap_threshold = 0.3);
};

#endif // FACEDETECTOR_H