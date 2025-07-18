#!/usr/bin/python
"""
<<<<<<< HEAD
Emotion Recognition Module
Uses a pretrained CNN model to classify emotions in face images.
=======
Optimized Emotion Recognition Module - Simplified DeepFace Version
Focuses on better face crop handling and minimal preprocessing for your dataset.
>>>>>>> deepface-version
"""

import cv2
import numpy as np
import os
import sys
import json
<<<<<<< HEAD
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
=======
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

# ANSI Color Codes for terminal output
class Colors:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    
    @staticmethod
    def success(text):
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    
    @staticmethod
    def error(text):
        return f"{Colors.RED}{text}{Colors.RESET}"
    
    @staticmethod
    def warning(text):
        return f"{Colors.YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def info(text):
        return f"{Colors.CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def bold(text):
        return f"{Colors.BOLD}{text}{Colors.RESET}"
>>>>>>> deepface-version

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
    def __init__(self, model_path="models/emotion_model.keras"):
        """
<<<<<<< HEAD
        Initialize the emotion recognizer with a pretrained model.
=======
        Initialize the emotion recognizer with optimized DeepFace configuration.
>>>>>>> deepface-version
        
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
        
<<<<<<< HEAD
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
=======
        print(Colors.success("Optimized DeepFace emotion recognition initialized!"))
        print(Colors.info("Note: Models will be downloaded on first use"))

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
                print(f"  {Colors.info('Resized from')} {width}x{height} {Colors.info('to')} {new_width}x{new_height}")
            
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
            print(Colors.warning(f"Warning: Enhancement failed, using original: {e}"))
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
                print(f"  {Colors.info('Trying strategy')} {i+1}: {Colors.info('enhance=')} {strategy['enhance']}, {Colors.info('model=')} {strategy['model']}")
                
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
                confidence = float(emotion_data[predicted_emotion_key] / 100.0)
                
                # Convert to standard format
                predicted_emotion = self.emotion_labels.get(predicted_emotion_key, predicted_emotion_key.capitalize())
                
                # Convert all probabilities to Python floats for JSON serialization
                all_probabilities = {
                    self.emotion_labels.get(key, key.capitalize()): float(value / 100.0) 
                    for key, value in emotion_data.items()
                }
                
                print(f"  {Colors.success('Strategy')} {i+1} {Colors.success('succeeded:')} {predicted_emotion} ({confidence:.3f})")
                return predicted_emotion, confidence, all_probabilities
                
            except Exception as e:
                print(f"  {Colors.error('Strategy')} {i+1} {Colors.error('failed:')} {str(e)[:50]}...")
                continue
        
        # If all strategies failed
        print(Colors.warning("  All strategies failed, returning neutral"))
        return "Neutral", 0.5, {emotion: 0.0 for emotion in self.emotion_labels.values()}

    def predict_emotion(self, face_image):
        """
        Main emotion prediction method.
>>>>>>> deepface-version
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted_emotion_string, confidence_score, all_probabilities)
        """
<<<<<<< HEAD
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
=======
        try:
            print(f"  {Colors.info('Input image shape:')} {face_image.shape}")
            return self.predict_emotion_robust(face_image)
            
        except Exception as e:
            print(Colors.error(f"Error in emotion prediction: {e}"))
            return "Neutral", 0.5, {emotion: 0.0 for emotion in self.emotion_labels.values()}
>>>>>>> deepface-version

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
        
        print(f"{Colors.info('Looking for face images with base filename:')} {base_filename}")
        
        # Process each detected face image
        while True:
            face_filename = f"{base_filename}_face_{face_index}.jpg"
            
            if not os.path.exists(face_filename):
                break
                
            print(f"{Colors.bold('Processing face image:')} {face_filename}")
            
            # Load face image
            face_image = cv2.imread(face_filename)
            if face_image is None:
                print(Colors.warning(f"Warning: Could not load {face_filename}"))
                face_index += 1
                continue
            
            # Predict emotion
            emotion, confidence, probabilities = self.predict_emotion(face_image)
            
            # Ensure all values are JSON serializable
            result = {
                'face_index': face_index,
                'face_filename': face_filename,
                'predicted_emotion': emotion,
                'confidence': float(confidence),  # Ensure it's a Python float
                'all_probabilities': {k: float(v) for k, v in probabilities.items()}  # Convert all to Python floats
            }
            
            results.append(result)
            
            print(f"{Colors.success('Face')} {face_index}: {Colors.bold(emotion)} {Colors.success('(confidence:')} {confidence:.3f}{Colors.success(')')}")
            
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
        
        print(f"{Colors.success('Emotions saved to:')} {emotions_file}")
        
        # Save detailed JSON results with proper serialization
        json_file = f"{base_filename}_emotion_results.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"{Colors.success('Detailed results saved to:')} {json_file}")
        except Exception as e:
            print(Colors.error(f"Error saving JSON results: {e}"))
            # Try to save with additional conversion
            try:
                # Convert any remaining numpy types to Python types
                serializable_results = []
                for result in results:
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, dict):
                            serializable_result[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                                      for k, v in value.items()}
                        elif isinstance(value, (np.floating, np.integer)):
                            serializable_result[key] = float(value)
                        else:
                            serializable_result[key] = value
                    serializable_results.append(serializable_result)
                
                with open(json_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"{Colors.success('Detailed results saved to:')} {json_file}")
            except Exception as e2:
                print(Colors.error(f"Failed to save JSON results: {e2}"))

def main():
    """
    Main function for standalone emotion recognition.
    """
    if len(sys.argv) < 2:
<<<<<<< HEAD
        print("Usage: python3 emotion_recognition.py <base_filename> [model_path]")
        print("Example: python3 emotion_recognition.py ../data/images/happy (1)")
        print("\nThis script processes face images created by the face detection module.")
        print("It looks for files like: <base_filename>_face_0.jpg, <base_filename>_face_1.jpg, etc.")
        return
    
    base_filename = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/emotion_model.keras"
=======
        print(Colors.bold("Usage:") + " python3 emotion_recognition.py <base_filename> [model_path]")
        print(Colors.bold("Example:") + " python3 emotion_recognition.py ../output/happy_1")
        print(f"\n{Colors.info('This script processes face images created by the face detection module.')}")
        print(f"{Colors.info('It looks for files like:')} <base_filename>_face_0.jpg, <base_filename>_face_1.jpg, etc.")
        print(f"\n{Colors.warning('Note:')} Using optimized DeepFace with robust fallback strategies.")
        return
    
    base_filename = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
>>>>>>> deepface-version
    
    try:
        # Initialize optimized emotion recognizer
        recognizer = EmotionRecognizer(model_path)
        
        # Process all face images
        results = recognizer.process_face_images(base_filename)
        
        if not results:
            print(Colors.warning("No face images found. Please run face detection first."))
            return
        
        # Save results
        recognizer.save_results(results, base_filename)
        
        # Print summary
<<<<<<< HEAD
        print(f"\nEmotion Recognition Summary:")
        print(f"Processed {len(results)} face(s)")
=======
        print(f"\n{Colors.bold('Optimized Emotion Recognition Summary:')}")
        print(f"{Colors.success('Processed')} {len(results)} {Colors.success('face(s)')}")
>>>>>>> deepface-version
        for result in results:
            print(f"{Colors.info('Face')} {result['face_index']}: {Colors.bold(result['predicted_emotion'])} "
                  f"{Colors.info('(confidence:')} {result['confidence']:.3f}{Colors.info(')')}")
        
    except Exception as e:
<<<<<<< HEAD
        print(f"Error during emotion recognition: {e}")
=======
        print(Colors.error(f"Error during emotion recognition: {e}"))
        print(Colors.warning("Make sure DeepFace is installed: pip install deepface"))
>>>>>>> deepface-version
        return 1

if __name__ == "__main__":
    main()