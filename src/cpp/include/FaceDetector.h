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
<<<<<<< HEAD

public:
    /**
     * Constructor: Initialize the face detector with cascade file
     * @param cascade_file Path to Haar cascade classifier XML file
     */
    FaceDetector(const std::string& cascade_file = "models/haarcascade_frontalface_alt.xml");

    /**
     * Main face detection method using Viola-Jones algorithm
     * @param image Input image to detect faces in
     * @return Vector of rectangles representing detected face regions
     */
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);

    /**
     * Save individual face regions as separate images
     * @param image Original image
     * @param faces Vector of face rectangles
     * @param base_filename Base filename for saving face images
     * @return Vector of saved face filenames
     */
    std::vector<std::string> saveFaceRegions(const cv::Mat& image, const std::vector<cv::Rect>& faces, const std::string& base_filename);

    /**
     * Draw bounding boxes around detected faces
     * @param image Original image
     * @param faces Vector of face rectangles
     * @return Image with face bounding boxes drawn
     */
    cv::Mat drawFaceBoxes(const cv::Mat& image, const std::vector<cv::Rect>& faces);

=======
    
    // Helper function for preprocessing
    cv::Mat preprocessImage(const cv::Mat& image);
    
    // Smart filtering to remove obvious false positives
    std::vector<cv::Rect> filterFalsePositives(const cv::Mat& image, const std::vector<cv::Rect>& faces);
    
    // Non-Maximum Suppression to remove overlapping detections
    std::vector<cv::Rect> applyNMS(const std::vector<cv::Rect>& faces, double overlap_threshold = 0.3);
>>>>>>> deepface-version
};

#endif // FACEDETECTOR_H