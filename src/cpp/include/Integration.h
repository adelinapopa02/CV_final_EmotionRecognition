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
<<<<<<< HEAD

public:
    /**
     * Constructor: Initialize the integrated system
     * @param cascade_file Path to Haar cascade classifier
     * @param python_script Path to emotion recognition Python script
     * @param emotion_model Path to emotion recognition model
     */
    Integration(
        const std::string& cascade_file = "models/haarcascade_frontalface_alt.xml",
        const std::string& python_script = "../src/python/emotion_recognition.py",
        const std::string& emotion_model = "models/emotion_model.keras"
    );

    /**
     * Run face detection only (standalone mode)
     * @param image_path Path to input image
     * @return True if successful, false otherwise
     */
    bool runFaceDetectionOnly(const std::string& image_path);

    /**
     * Run complete pipeline (face detection + emotion recognition)
     * @param image_path Path to input image
     * @return True if successful, false otherwise
     */
    bool runCompleteIntegration(const std::string& image_path);

private:
    /**
     * Run emotion recognition using Python script
     * @param base_filename Base filename for face images
     * @return Vector of emotion predictions
     */
=======
    
>>>>>>> deepface-version
    std::vector<std::string> runEmotionRecognition(const std::string& base_filename);
    void saveUnifiedResults(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions, const std::string& base_filename);
    cv::Mat createFinalResult(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::string>& emotions);
    std::string getBaseFilename(const std::string& filepath);
};

#endif // INTEGRATION_H