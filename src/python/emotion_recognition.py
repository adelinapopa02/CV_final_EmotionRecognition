#!/usr/bin/python
"""
Emotion Recognition Module - DeepFace Version
Uses DeepFace library with multiple pre-trained models for better emotion classification.
"""

import cv2
import numpy as np
import os
import sys
import json
from deepface import DeepFace

class EmotionRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer with DeepFace.
        
        Args:
            model_path (str): Not used with DeepFace (kept for compatibility)
        """
        # DeepFace emotion labels (standardized)
        self.emotion_labels = {
            'angry': 'Angry',
            'disgust': 'Disgust', 
            'fear': 'Fear',
            'happy': 'Happy',
            'sad': 'Sad',
            'surprise': 'Surprise',
            'neutral': 'Neutral'
        }
        
        print("DeepFace emotion recognition initialized!")
        print("Note: Models will be downloaded on first use (may take a moment)")

    def predict_emotion(self, face_image):
        """
        Predict emotion for a single face image using DeepFace.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted_emotion_string, confidence_score, all_probabilities)
        """
        try:
            # DeepFace expects RGB images, OpenCV loads as BGR
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Use DeepFace to analyze emotion
            # enforce_detection=False allows processing even if face detection fails
            result = DeepFace.analyze(
                img_path=rgb_image, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True,  # Suppress verbose output
                detector_backend='opencv'  # Use OpenCV backend instead of default
            )
            
            # DeepFace returns a list, take first result
            if isinstance(result, list):
                emotion_data = result[0]['emotion']
            else:
                emotion_data = result['emotion']
            
            # Find the emotion with highest confidence
            predicted_emotion_key = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[predicted_emotion_key] / 100.0  # Convert percentage to 0-1
            
            # Convert to our standard label format
            predicted_emotion = self.emotion_labels.get(predicted_emotion_key, predicted_emotion_key.capitalize())
            
            # Convert all probabilities to our format
            all_probabilities = {
                self.emotion_labels.get(key, key.capitalize()): value / 100.0 
                for key, value in emotion_data.items()
            }
            
            return predicted_emotion, confidence, all_probabilities
            
        except Exception as e:
            print(f"Error in DeepFace emotion prediction: {e}")
            # Fallback to neutral if prediction fails
            return "Neutral", 0.5, {emotion: 0.0 for emotion in self.emotion_labels.values()}

    def process_face_images(self, base_filename):
        """
        Process all face images extracted by the face detection module.
        
        Args:
            base_filename (str): Base filename used by face detection module
            
        Returns:
            list: List of emotion predictions for each face
        """
        results = []
        face_index = 0
        
        print(f"Looking for face images with base filename: {base_filename}")
        
        # Process each detected face image
        while True:
            face_filename = f"{base_filename}_face_{face_index}.jpg"
            
            if not os.path.exists(face_filename):
                break
                
            print(f"Processing face image: {face_filename}")
            
            # Load face image
            face_image = cv2.imread(face_filename)
            if face_image is None:
                print(f"Warning: Could not load {face_filename}")
                face_index += 1
                continue
            
            # Predict emotion
            emotion, confidence, probabilities = self.predict_emotion(face_image)
            
            result = {
                'face_index': face_index,
                'face_filename': face_filename,
                'predicted_emotion': emotion,
                'confidence': confidence,
                'all_probabilities': probabilities
            }
            
            results.append(result)
            
            print(f"Face {face_index}: {emotion} (confidence: {confidence:.3f})")
            
            face_index += 1
        
        return results

    def save_results(self, results, base_filename):
        """
        Save emotion recognition results to files.
        
        Args:
            results (list): List of emotion prediction results
            base_filename (str): Base filename for output files
        """
        # Save simple text file with just emotions
        emotions_file = f"{base_filename}_emotions.txt"
        with open(emotions_file, 'w') as f:
            for result in results:
                f.write(f"{result['predicted_emotion']}\n")
        
        print(f"Emotions saved to: {emotions_file}")
        
        # Save detailed JSON results
        json_file = f"{base_filename}_emotion_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to: {json_file}")

def main():
    """
    Main function for standalone emotion recognition.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 emotion_recognition.py <base_filename> [model_path]")
        print("Example: python3 emotion_recognition.py ../output/happy_1")
        print("\nThis script processes face images created by the face detection module.")
        print("It looks for files like: <base_filename>_face_0.jpg, <base_filename>_face_1.jpg, etc.")
        print("\nNote: Using DeepFace - models will be downloaded automatically on first use.")
        return
    
    base_filename = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None  # Not used with DeepFace
    
    try:
        # Initialize emotion recognizer
        recognizer = EmotionRecognizer(model_path)
        
        # Process all face images
        results = recognizer.process_face_images(base_filename)
        
        if not results:
            print("No face images found. Please run face detection first.")
            return
        
        # Save results
        recognizer.save_results(results, base_filename)
        
        # Print summary
        print(f"\nEmotion Recognition Summary (DeepFace):")
        print(f"Processed {len(results)} face(s)")
        for result in results:
            print(f"Face {result['face_index']}: {result['predicted_emotion']} "
                  f"(confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"Error during emotion recognition: {e}")
        print("Make sure DeepFace is installed: pip install deepface")
        return 1

if __name__ == "__main__":
    main()