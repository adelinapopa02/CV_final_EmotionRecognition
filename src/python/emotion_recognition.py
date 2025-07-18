#!/usr/bin/python
"""
Optimized Emotion Recognition Module - Simplified DeepFace Version
Focuses on better face crop handling and minimal preprocessing for your dataset.
"""

import cv2
import numpy as np
import os
import sys
import json
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer with optimized DeepFace configuration.
        
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
        
        print("Optimized DeepFace emotion recognition initialized!")
        print("Note: Models will be downloaded on first use")

    def enhance_face_crop(self, face_image):
        """
        Minimal but effective enhancement for face crops.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            numpy.ndarray: Enhanced face image
        """
        try:
            # Ensure we have a color image
            if len(face_image.shape) == 2:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif len(face_image.shape) == 3 and face_image.shape[2] == 3:  # BGR
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Only resize if the image is very small (keep original size when possible)
            height, width = face_image.shape[:2]
            min_size = 48  # DeepFace minimum
            
            if height < min_size or width < min_size:
                # Calculate scale factor to reach minimum size
                scale = max(min_size / height, min_size / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                face_image = cv2.resize(face_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"  Resized from {width}x{height} to {new_width}x{new_height}")
            
            # Very light enhancement - just improve contrast slightly
            # Convert to float for processing
            img_float = face_image.astype(np.float32) / 255.0
            
            # Gentle contrast enhancement using gamma correction
            gamma = 0.9  # Slightly brighten dark areas
            enhanced = np.power(img_float, gamma)
            
            # Convert back to uint8
            face_image = (enhanced * 255).astype(np.uint8)
            
            return face_image
            
        except Exception as e:
            print(f"Warning: Enhancement failed, using original: {e}")
            return face_image

    def predict_emotion_robust(self, face_image):
        """
        Robust emotion prediction with multiple fallback strategies.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted_emotion_string, confidence_score, all_probabilities)
        """
        strategies = [
            # Strategy 1: Enhanced image with VGG-Face
            {'enhance': True, 'model': 'VGG-Face', 'detector': 'opencv'},
            # Strategy 2: Original image with VGG-Face
            {'enhance': False, 'model': 'VGG-Face', 'detector': 'opencv'},
            # Strategy 3: Enhanced image with default model
            {'enhance': True, 'model': None, 'detector': 'opencv'},
            # Strategy 4: Original image with default model
            {'enhance': False, 'model': None, 'detector': 'opencv'},
            # Strategy 5: Last resort - skip face detection entirely
            {'enhance': True, 'model': None, 'detector': None}
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"  Trying strategy {i+1}: enhance={strategy['enhance']}, model={strategy['model']}")
                
                # Prepare image
                if strategy['enhance']:
                    img_to_use = self.enhance_face_crop(face_image)
                else:
                    # Minimal conversion for original image
                    if len(face_image.shape) == 3:
                        img_to_use = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    else:
                        img_to_use = face_image
                
                # Prepare DeepFace parameters
                analyze_params = {
                    'img_path': img_to_use,
                    'actions': ['emotion'],
                    'enforce_detection': False,
                    'silent': True
                }
                
                # Add model if specified
                if strategy['model']:
                    analyze_params['model_name'] = strategy['model']
                
                # Add detector if specified
                if strategy['detector']:
                    analyze_params['detector_backend'] = strategy['detector']
                
                # Try the analysis
                result = DeepFace.analyze(**analyze_params)
                
                # Extract emotion data
                if isinstance(result, list):
                    emotion_data = result[0]['emotion']
                else:
                    emotion_data = result['emotion']
                
                # Find best emotion
                predicted_emotion_key = max(emotion_data, key=emotion_data.get)
                confidence = emotion_data[predicted_emotion_key] / 100.0
                
                # Convert to standard format
                predicted_emotion = self.emotion_labels.get(predicted_emotion_key, predicted_emotion_key.capitalize())
                
                all_probabilities = {
                    self.emotion_labels.get(key, key.capitalize()): value / 100.0 
                    for key, value in emotion_data.items()
                }
                
                print(f"  Strategy {i+1} succeeded: {predicted_emotion} ({confidence:.3f})")
                return predicted_emotion, confidence, all_probabilities
                
            except Exception as e:
                print(f"  Strategy {i+1} failed: {str(e)[:50]}...")
                continue
        
        # If all strategies failed
        print("  All strategies failed, returning neutral")
        return "Neutral", 0.5, {emotion: 0.0 for emotion in self.emotion_labels.values()}

    def predict_emotion(self, face_image):
        """
        Main emotion prediction method.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted_emotion_string, confidence_score, all_probabilities)
        """
        try:
            print(f"  Input image shape: {face_image.shape}")
            return self.predict_emotion_robust(face_image)
            
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
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
        print("\nNote: Using optimized DeepFace with robust fallback strategies.")
        return
    
    base_filename = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize optimized emotion recognizer
        recognizer = EmotionRecognizer(model_path)
        
        # Process all face images
        results = recognizer.process_face_images(base_filename)
        
        if not results:
            print("No face images found. Please run face detection first.")
            return
        
        # Save results
        recognizer.save_results(results, base_filename)
        
        # Print summary
        print(f"\nOptimized Emotion Recognition Summary:")
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