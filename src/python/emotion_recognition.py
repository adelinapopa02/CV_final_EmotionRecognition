#!/usr/bin/python
"""
Emotion Recognition Module
Uses a pretrained CNN model to classify emotions in face images.
"""

import cv2
import numpy as np
import os
import sys
import json
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")   
class SEBlock(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.squeeze = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        se = self.squeeze(inputs)               
        se = self.fc1(se)                      
        se = self.fc2(se)                      
        se = tf.reshape(se, [-1, 1, 1, self.channels])
        return inputs * se                     

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config
    
class EmotionRecognizer:
    def __init__(self, model_path="../models/emotion_model.keras"):
        """
        Initialize the emotion recognizer with a pretrained model.
        
        Args:
            model_path (str): Path to the pretrained emotion recognition model
        """
        self.model_path = model_path
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Load the pretrained model
        try:
            self.model = load_model(model_path, custom_objects = {'SEBlock': SEBlock})
            print(f"Emotion recognition model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Please ensure the emotion_model.keras file is in the models directory")
            raise

    def preprocess_face(self, face_image):
        """
        Preprocess face image for emotion recognition.
        The FER-2013 model expects 48x48 grayscale images.
        
        Args:
            face_image (numpy.ndarray): Input face image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model prediction
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image.copy()
        
        # Resize to 48x48 (FER-2013 model input size)
        resized_face = cv2.resize(gray_face, (224, 224))
        
        # Normalize pixel values to [0, 1]
        normalized_face = resized_face.astype('float32') / 255.0
        
        # Reshape for model input: (1, 224, 224, 1)
        # 1 = batch size, 224x224 = image dimensions, 1 = grayscale channel
        preprocessed_face = normalized_face.reshape(1, 224, 224, 1)
        
        return preprocessed_face

    def predict_emotion(self, face_image):
        """
        Predict emotion for a single face image.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted_emotion_string, confidence_score, all_probabilities)
        """
        # Preprocess the face image
        preprocessed_face = self.preprocess_face(face_image)
        
        # Make prediction using the CNN model
        predictions = self.model.predict(preprocessed_face, verbose=0)
        
        # Get the class with highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Convert to emotion label
        predicted_emotion = self.emotion_labels[predicted_class]
        
        # Get all probabilities for detailed analysis
        all_probabilities = {
            self.emotion_labels[i]: float(predictions[0][i]) 
            for i in range(len(self.emotion_labels))
        }
        
        return predicted_emotion, confidence, all_probabilities

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
        print("Example: python3 emotion_recognition.py ../data/images/happy (1)")
        print("\nThis script processes face images created by the face detection module.")
        print("It looks for files like: <base_filename>_face_0.jpg, <base_filename>_face_1.jpg, etc.")
        return
    
    base_filename = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "../models/emotion_model.keras"
    
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
        print(f"\nEmotion Recognition Summary:")
        print(f"Processed {len(results)} face(s)")
        for result in results:
            print(f"Face {result['face_index']}: {result['predicted_emotion']} "
                  f"(confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"Error during emotion recognition: {e}")
        return 1

if __name__ == "__main__":
    main()