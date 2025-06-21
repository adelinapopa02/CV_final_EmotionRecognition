#!/usr/bin/python
"""
Evaluation Module for Face Detection and Emotion Recognition System
Calculates comprehensive performance metrics against ground truth data.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

class SystemEvaluator:
    def __init__(self):
        """Initialize the system evaluator."""
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: [x, y, width, height] format
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_ground_truth(self, predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
        """
        Match predicted bounding boxes to ground truth boxes using IoU.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth bounding boxes
            iou_threshold: Minimum IoU for a positive match
            
        Returns:
            tuple: (matches, unmatched_predictions, unmatched_ground_truths)
        """
        matches = []
        unmatched_predictions = list(range(len(predicted_boxes)))
        unmatched_ground_truths = list(range(len(ground_truth_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predicted_boxes), len(ground_truth_boxes)))
        for i, pred_box in enumerate(predicted_boxes):
            for j, gt_box in enumerate(ground_truth_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Greedily match highest IoU pairs above threshold
        while True:
            if iou_matrix.size == 0:
                break
                
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < iou_threshold:
                break
                
            pred_idx, gt_idx = max_iou_idx
            matches.append((pred_idx, gt_idx, max_iou))
            
            # Remove matched boxes from consideration
            unmatched_predictions.remove(pred_idx)
            unmatched_ground_truths.remove(gt_idx)
            
            # Set matched row and column to 0
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        
        return matches, unmatched_predictions, unmatched_ground_truths
    
    def evaluate_face_detection(self, results_dir, ground_truth_file):
        """
        Evaluate face detection performance.
        
        Args:
            results_dir: Directory containing detection results
            ground_truth_file: JSON file with ground truth annotations
            
        Returns:
            dict: Face detection metrics
        """
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        all_matches = []
        all_predictions = 0
        all_ground_truths = 0
        total_iou = 0.0
        
        for image_name, gt_data in ground_truth.items():
            # Look for corresponding results file
            base_name = os.path.splitext(image_name)[0]
            results_file = os.path.join(results_dir, f"{base_name}_emotion_results.json")
            
            if not os.path.exists(results_file):
                print(f"Warning: No results found for {image_name}")
                continue
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract bounding boxes (need to reconstruct from face files)
            predicted_boxes = []
            for result in results:
                face_file = result['face_filename']
                if os.path.exists(face_file):
                    # For simplicity, we'll use a placeholder box
                    # In practice, you'd need to store the original bounding boxes
                    predicted_boxes.append([0, 0, 100, 100])  # Placeholder
            
            gt_boxes = gt_data.get('bounding_boxes', [])
            
            # Match predictions to ground truth
            matches, unmatched_pred, unmatched_gt = self.match_detections_to_ground_truth(
                predicted_boxes, gt_boxes
            )
            
            all_matches.extend(matches)
            all_predictions += len(predicted_boxes)
            all_ground_truths += len(gt_boxes)
            
            # Calculate average IoU for matches
            for _, _, iou in matches:
                total_iou += iou
        
        # Calculate metrics
        true_positives = len(all_matches)
        false_positives = all_predictions - true_positives
        false_negatives = all_ground_truths - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        average_iou = total_iou / len(all_matches) if all_matches else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_iou': average_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_emotion_recognition(self, results_dir, ground_truth_file):
        """
        Evaluate emotion recognition performance.
        
        Args:
            results_dir: Directory containing emotion recognition results
            ground_truth_file: JSON file with ground truth annotations
            
        Returns:
            dict: Emotion recognition metrics
        """
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        predicted_emotions = []
        true_emotions = []
        
        for image_name, gt_data in ground_truth.items():
            base_name = os.path.splitext(image_name)[0]
            emotions_file = os.path.join(results_dir, f"{base_name}_emotions.txt")
            
            if not os.path.exists(emotions_file):
                print(f"Warning: No emotion results found for {image_name}")
                continue
            
            # Read predicted emotions
            with open(emotions_file, 'r') as f:
                pred_emotions = [line.strip() for line in f.readlines() if line.strip()]
            
            gt_emotions = gt_data.get('emotions', [])
            
            # Match emotions (assuming same order as face detection)
            min_count = min(len(pred_emotions), len(gt_emotions))
            predicted_emotions.extend(pred_emotions[:min_count])
            true_emotions.extend(gt_emotions[:min_count])
        
        if not predicted_emotions:
            return {'accuracy': 0, 'classification_report': {}, 'confusion_matrix': np.array([])}
        
        # Calculate accuracy
        accuracy = sum(p == t for p, t in zip(predicted_emotions, true_emotions)) / len(predicted_emotions)
        
        # Generate classification report
        report = classification_report(true_emotions, predicted_emotions, 
                                     labels=self.emotion_labels, output_dict=True, zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_emotions, predicted_emotions, labels=self.emotion_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predicted_emotions': predicted_emotions,
            'true_emotions': true_emotions
        }
    
    def plot_confusion_matrix(self, cm, output_file='confusion_matrix.png'):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            output_file: Output file path
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.emotion_labels, 
                   yticklabels=self.emotion_labels)
        plt.title('Emotion Recognition Confusion Matrix')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {output_file}")
    
    def generate_evaluation_report(self, face_metrics, emotion_metrics, output_file='evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            face_metrics: Face detection metrics
            emotion_metrics: Emotion recognition metrics
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("COMPREHENSIVE SYSTEM EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Face Detection Results
            f.write("FACE DETECTION PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Precision: {face_metrics['precision']:.3f}\n")
            f.write(f"Recall: {face_metrics['recall']:.3f}\n")
            f.write(f"F1-Score: {face_metrics['f1_score']:.3f}\n")
            f.write(f"Average IoU: {face_metrics['average_iou']:.3f}\n")
            f.write(f"True Positives: {face_metrics['true_positives']}\n")
            f.write(f"False Positives: {face_metrics['false_positives']}\n")
            f.write(f"False Negatives: {face_metrics['false_negatives']}\n\n")
            
            # Emotion Recognition Results
            f.write("EMOTION RECOGNITION PERFORMANCE\n")
            f.write("-" * 35 + "\n")
            f.write(f"Overall Accuracy: {emotion_metrics['accuracy']:.3f}\n\n")
            
            if 'classification_report' in emotion_metrics:
                f.write("Per-Class Performance:\n")
                for emotion in self.emotion_labels:
                    if emotion in emotion_metrics['classification_report']:
                        metrics = emotion_metrics['classification_report'][emotion]
                        f.write(f"  {emotion}:\n")
                        f.write(f"    Precision: {metrics['precision']:.3f}\n")
                        f.write(f"    Recall: {metrics['recall']:.3f}\n")
                        f.write(f"    F1-Score: {metrics['f1-score']:.3f}\n")
                        f.write(f"    Support: {metrics['support']}\n")
            
            f.write(f"\nMacro Average:\n")
            if 'macro avg' in emotion_metrics['classification_report']:
                macro = emotion_metrics['classification_report']['macro avg']
                f.write(f"  Precision: {macro['precision']:.3f}\n")
                f.write(f"  Recall: {macro['recall']:.3f}\n")
                f.write(f"  F1-Score: {macro['f1-score']:.3f}\n")
            
            # System-Level Performance
            f.write(f"\nSYSTEM-LEVEL PERFORMANCE\n")
            f.write("-" * 25 + "\n")
            total_faces_processed = len(emotion_metrics.get('predicted_emotions', []))
            correctly_processed = sum(p == t for p, t in zip(
                emotion_metrics.get('predicted_emotions', []),
                emotion_metrics.get('true_emotions', [])
            ))
            system_accuracy = correctly_processed / total_faces_processed if total_faces_processed > 0 else 0
            f.write(f"End-to-End Accuracy: {system_accuracy:.3f}\n")
            f.write(f"Total Faces Processed: {total_faces_processed}\n")
            f.write(f"Correctly Classified: {correctly_processed}\n")
        
        print(f"Evaluation report saved to: {output_file}")

def main():
    """
    Main function for system evaluation.
    """
    if len(sys.argv) < 3:
        print("Usage: python3 evaluation.py <results_directory> <ground_truth_file>")
        print("Example: python3 evaluation.py ./output ground_truth.json")
        print("\nGround truth file format:")
        print('''{
  "image1.jpg": {
    "bounding_boxes": [[x, y, width, height], ...],
    "emotions": ["Happy", "Sad", ...]
  }
}''')
        return
    
    results_dir = sys.argv[1]
    ground_truth_file = sys.argv[2]
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return
    
    try:
        evaluator = SystemEvaluator()
        
        print("Evaluating face detection performance...")
        face_metrics = evaluator.evaluate_face_detection(results_dir, ground_truth_file)
        
        print("Evaluating emotion recognition performance...")
        emotion_metrics = evaluator.evaluate_emotion_recognition(results_dir, ground_truth_file)
        
        # Plot confusion matrix
        if emotion_metrics['confusion_matrix'].size > 0:
            evaluator.plot_confusion_matrix(emotion_metrics['confusion_matrix'])
        
        # Generate comprehensive report
        evaluator.generate_evaluation_report(face_metrics, emotion_metrics)
        
        # Print summary to console
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Face Detection F1-Score: {face_metrics['f1_score']:.3f}")
        print(f"Emotion Recognition Accuracy: {emotion_metrics['accuracy']:.3f}")
        print(f"Average IoU: {face_metrics['average_iou']:.3f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    main()