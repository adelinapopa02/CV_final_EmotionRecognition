#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "FaceDetector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Integration {
public:
    Integration(const std::string& cascade_file, const std::string& python_script, const std::string& emotion_model);
    
    bool runFaceDetectionOnly(const std::string& image_path);
    bool runCompleteIntegration(const std::string& image_path);

private:
    FaceDetector face_detector;
    std::string python_script_path;
    std::string model_path;
    
    std::vector<std::string> runEmotionRecognition(const std::string& base_filename);
    void saveUnifiedResults(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions, const std::string& base_filename);
    cv::Mat createFinalResult(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions);
    std::string getBaseFilename(const std::string& filepath);
};

#endif // INTEGRATION_H