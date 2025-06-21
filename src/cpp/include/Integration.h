#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "FaceDetector.h"
#include <string>
#include <vector>

/**
 * SystemIntegrator Class
 * 
 * Handles the complete face detection + emotion recognition pipeline.
 * Orchestrates the integration between C++ face detection and Python emotion recognition.
 */
class Integration {
private:
    FaceDetector face_detector;
    std::string python_script_path;
    std::string model_path;

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
    std::vector<std::string> runEmotionRecognition(const std::string& base_filename);

    /**
     * Create final annotated result with face boxes and emotion labels
     * @param image Original image
     * @param faces Vector of face rectangles
     * @param emotions Vector of emotion labels
     * @return Annotated image
     */
    cv::Mat createFinalResult(const cv::Mat& image, 
                             const std::vector<cv::Rect>& faces, 
                             const std::vector<std::string>& emotions);

    /**
     * Extract base filename from full path
     * @param filepath Full path to file
     * @return Base filename without extension
     */
    std::string getBaseFilename(const std::string& filepath);
};

#endif // INTEGRATION_H