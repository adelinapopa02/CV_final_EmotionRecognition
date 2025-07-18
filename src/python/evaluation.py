#!/usr/bin/python
"""
Simplified Evaluation Module for Face Detection and Emotion Recognition System
Directly compares unified results with ground truth labels.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class SystemEvaluator:
    def __init__(self):
        """Initialize the system evaluator."""
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.class_to_emotion = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear', 
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes in [x, y, w, h] format."""
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
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def parse_yolo_file(self, file_path, img_width=1280, img_height=853):
        """Parse YOLO format file and return boxes and emotions."""
        boxes = []
        emotions = []
        
        if not os.path.exists(file_path):
            return boxes, emotions
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()  # Read all content and strip whitespace
            
            if not content:  # File is empty
                return boxes, emotions
                
            lines = content.split('\n')  # Split by newlines
            
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) == 5:
                        emotion_class, center_x, center_y, width, height = map(float, parts)
                        
                        # Convert from YOLO to [x, y, width, height] format
                        x = int((center_x - width/2) * img_width)
                        y = int((center_y - height/2) * img_height)
                        w = int(width * img_width)
                        h = int(height * img_height)
                        
                        boxes.append([x, y, w, h])
                        emotions.append(self.class_to_emotion.get(int(emotion_class), 'Neutral'))
                        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return boxes, emotions
    
    def match_faces(self, pred_boxes, pred_emotions, gt_boxes, gt_emotions, iou_threshold=0.5):
        """Match predicted faces to ground truth faces using IoU."""
        matches = []
        unmatched_pred = list(range(len(pred_boxes)))
        unmatched_gt = list(range(len(gt_boxes)))
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return matches, unmatched_pred, unmatched_gt
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
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
            matches.append((pred_idx, gt_idx, max_iou, 
                          pred_emotions[pred_idx], gt_emotions[gt_idx]))
            
            unmatched_pred.remove(pred_idx)
            unmatched_gt.remove(gt_idx)
            
            # Set matched row and column to 0
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        
        return matches, unmatched_pred, unmatched_gt
    
    def evaluate_system(self, results_dir, labels_dir):
        """Evaluate the system by comparing results with ground truth labels."""
        
        # Find all unified results files
        result_files = []
        for file in os.listdir(results_dir):
            if file.endswith('_unified_results.txt'):
                result_files.append(file)
        
        if not result_files:
            print("Error: No unified results files found in results directory")
            return None, None
        
        print(f"Found {len(result_files)} result files to evaluate")
        
        # Initialize metrics
        all_matches = []
        all_pred_emotions = []
        all_gt_emotions = []
        total_predictions = 0
        total_ground_truth = 0
        total_iou = 0.0
        
        processed_files = 0
        
        for result_file in result_files:
            # Extract base name: angry_1_unified_results.txt -> angry_1
            base_name = result_file.replace('_unified_results.txt', '')
            
            # Look for corresponding label file
            # Handle different naming conventions
            possible_label_files = [
                f"{base_name}.txt",
                f"{base_name.replace('_', ' ')}.txt",
                f"{base_name.replace('_', ' (')}.txt"
            ]
            
            label_file = None
            for possible_file in possible_label_files:
                if os.path.exists(os.path.join(labels_dir, possible_file)):
                    label_file = os.path.join(labels_dir, possible_file)
                    break
            
            if not label_file:
                print(f"Warning: No label file found for {base_name}")
                continue
            
            # Parse files
            result_path = os.path.join(results_dir, result_file)
            pred_boxes, pred_emotions = self.parse_yolo_file(result_path)
            gt_boxes, gt_emotions = self.parse_yolo_file(label_file)
            
            if not pred_boxes and not gt_boxes:
                continue
            
            print(f"Processing {base_name}: {len(pred_boxes)} predicted, {len(gt_boxes)} ground truth")
            
            # Match faces
            matches, unmatched_pred, unmatched_gt = self.match_faces(
                pred_boxes, pred_emotions, gt_boxes, gt_emotions
            )
            
            all_matches.extend(matches)
            total_predictions += len(pred_boxes)
            total_ground_truth += len(gt_boxes)
            
            # Collect emotion predictions for matched faces
            for match in matches:
                pred_idx, gt_idx, iou, pred_emotion, gt_emotion = match
                all_pred_emotions.append(pred_emotion)
                all_gt_emotions.append(gt_emotion)
                total_iou += iou
            
            processed_files += 1
        
        if processed_files == 0:
            print("Error: No files could be processed")
            return None, None
        
        # Calculate face detection metrics
        true_positives = len(all_matches)
        false_positives = total_predictions - true_positives
        false_negatives = total_ground_truth - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = total_iou / len(all_matches) if all_matches else 0
        
        face_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_iou': avg_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        # Calculate emotion recognition metrics
        emotion_metrics = {}
        if all_pred_emotions:
            accuracy = sum(p == g for p, g in zip(all_pred_emotions, all_gt_emotions)) / len(all_pred_emotions)
            
            # Generate classification report
            report = classification_report(all_gt_emotions, all_pred_emotions, 
                                         labels=self.emotion_labels, output_dict=True, zero_division=0)
            
            # Generate confusion matrix
            cm = confusion_matrix(all_gt_emotions, all_pred_emotions, labels=self.emotion_labels)
            
            emotion_metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predicted_emotions': all_pred_emotions,
                'true_emotions': all_gt_emotions
            }
        
        print(f"\nProcessed {processed_files} files successfully")
        return face_metrics, emotion_metrics
    
    def plot_confusion_matrix(self, cm, output_file='confusion_matrix.png'):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.emotion_labels, 
                   yticklabels=self.emotion_labels)
        plt.title('Emotion Recognition Confusion Matrix (DeepFace)')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {output_file}")
    
    def generate_report(self, face_metrics, emotion_metrics, output_file='evaluation_report.txt'):
        """Generate comprehensive evaluation report."""
        with open(output_file, 'w') as f:
            f.write("FACE DETECTION AND EMOTION RECOGNITION EVALUATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FACE DETECTION PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Precision: {face_metrics['precision']:.3f}\n")
            f.write(f"Recall: {face_metrics['recall']:.3f}\n")
            f.write(f"F1-Score: {face_metrics['f1_score']:.3f}\n")
            f.write(f"Average IoU: {face_metrics['average_iou']:.3f}\n")
            f.write(f"True Positives: {face_metrics['true_positives']}\n")
            f.write(f"False Positives: {face_metrics['false_positives']}\n")
            f.write(f"False Negatives: {face_metrics['false_negatives']}\n\n")
            
            f.write("EMOTION RECOGNITION PERFORMANCE (DeepFace)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {emotion_metrics['accuracy']:.3f}\n\n")
            
            f.write("Per-Class Performance:\n")
            for emotion in self.emotion_labels:
                if emotion in emotion_metrics['classification_report']:
                    metrics = emotion_metrics['classification_report'][emotion]
                    f.write(f"  {emotion}:\n")
                    f.write(f"    Precision: {metrics['precision']:.3f}\n")
                    f.write(f"    Recall: {metrics['recall']:.3f}\n")
                    f.write(f"    F1-Score: {metrics['f1-score']:.3f}\n")
                    f.write(f"    Support: {metrics['support']}\n")
            
            if 'macro avg' in emotion_metrics['classification_report']:
                macro = emotion_metrics['classification_report']['macro avg']
                f.write(f"\nMacro Average:\n")
                f.write(f"  Precision: {macro['precision']:.3f}\n")
                f.write(f"  Recall: {macro['recall']:.3f}\n")
                f.write(f"  F1-Score: {macro['f1-score']:.3f}\n")
        
        print(f"Evaluation report saved to: {output_file}")

def main():
    """Main function for system evaluation."""
    if len(sys.argv) < 3:
        print("Usage: python3 evaluation.py <results_directory> <labels_directory>")
        print("Example: python3 evaluation.py ./output ./data/labels")
        return
    
    results_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    try:
        evaluator = SystemEvaluator()
        
        print("Evaluating system by comparing unified results with ground truth labels...")
        face_metrics, emotion_metrics = evaluator.evaluate_system(results_dir, labels_dir)
        
        if face_metrics is None or emotion_metrics is None:
            print("Evaluation failed!")
            return
        
        # Plot confusion matrix
        if 'confusion_matrix' in emotion_metrics and emotion_metrics['confusion_matrix'].size > 0:
            evaluator.plot_confusion_matrix(emotion_metrics['confusion_matrix'])
        
        # Generate report
        evaluator.generate_report(face_metrics, emotion_metrics)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Face Detection F1-Score: {face_metrics['f1_score']:.3f}")
        print(f"Face Detection Average IoU: {face_metrics['average_iou']:.3f}")
        print(f"Emotion Recognition Accuracy: {emotion_metrics['accuracy']:.3f}")
        print(f"Total Faces Evaluated: {len(emotion_metrics.get('predicted_emotions', []))}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()