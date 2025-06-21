#!/usr/bin/env python3
"""
Create Ground Truth JSON from Image Names and Label Files
=========================================================

This script automatically generates a ground_truth.json file by:
1. Extracting emotion from image filenames (e.g., "happy_1.jpg" -> "Happy")
2. Reading bounding boxes from corresponding label files in the labels folder

Usage:
    python3 create_ground_truth.py data/images data/labels
"""

import os
import sys
import json
import glob

def extract_emotion_from_filename(filename):
    """
    Extract emotion from filename like "happy (1).jpg" -> "Happy"
    """
    emotion_mapping = {
        'happy': 'Happy',
        'sad': 'Sad', 
        'angry': 'Angry',
        'fear': 'Fear',
        'surprise': 'Surprise',
        'disgust': 'Disgust',
        'neutral': 'Neutral'
    }
    
    # Get filename without extension
    base_name = os.path.splitext(filename)[0].lower()
    
    # Handle format like "happy (1)" or "happy(1)"
    # Split by space or opening parenthesis to get the emotion part
    import re
    emotion_part = re.split(r'[\s(]', base_name)[0]
    
    # Look up the emotion
    if emotion_part in emotion_mapping:
        return emotion_mapping[emotion_part]
    
    # If no exact match, try partial matching
    for emotion_key, emotion_label in emotion_mapping.items():
        if emotion_part.startswith(emotion_key):
            return emotion_label
    
    # If no emotion found, try to guess from the filename
    print(f"Warning: Could not extract emotion from filename '{filename}'")
    print(f"Expected format: 'emotion (number).jpg' where emotion is one of: {list(emotion_mapping.keys())}")
    return "Unknown"

def parse_label_file(label_file_path, image_width=None, image_height=None):
    """
    Parse label file to extract bounding boxes.
    
    Common formats:
    1. YOLO format: class_id center_x center_y width height (normalized 0-1)
    2. Pascal VOC format: xmin ymin xmax ymax (absolute coordinates)
    3. Custom format: x y width height (absolute coordinates)
    
    Args:
        label_file_path: Path to the label file
        image_width: Image width (needed for YOLO format conversion)
        image_height: Image height (needed for YOLO format conversion)
    
    Returns:
        List of bounding boxes in format [x, y, width, height]
    """
    bounding_boxes = []
    
    if not os.path.exists(label_file_path):
        print(f"Warning: Label file not found: {label_file_path}")
        return bounding_boxes
    
    try:
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            parts = line.split()
            
            if len(parts) == 5:
                # YOLO format: class_id center_x center_y width height (normalized)
                print(f"Detected YOLO format in {label_file_path}")
                if image_width is None or image_height is None:
                    print(f"Warning: YOLO format requires image dimensions. Skipping line {line_num}")
                    continue
                
                class_id, center_x, center_y, width, height = map(float, parts)
                
                # Convert from YOLO to [x, y, width, height] format
                x = int((center_x - width/2) * image_width)
                y = int((center_y - height/2) * image_height)
                w = int(width * image_width)
                h = int(height * image_height)
                
                bounding_boxes.append([x, y, w, h])
                
            elif len(parts) == 4:
                # Two possible formats:
                # 1. Pascal VOC: xmin ymin xmax ymax
                # 2. Custom: x y width height
                
                nums = list(map(float, parts))
                
                # Heuristic to detect format:
                # If 3rd and 4th values are much larger than 1st and 2nd, likely Pascal VOC
                if nums[2] > nums[0] and nums[3] > nums[1] and nums[2] > 100 and nums[3] > 100:
                    # Pascal VOC format: xmin ymin xmax ymax
                    print(f"Detected Pascal VOC format in {label_file_path}")
                    xmin, ymin, xmax, ymax = map(int, nums)
                    x = xmin
                    y = ymin
                    width = xmax - xmin
                    height = ymax - ymin
                    bounding_boxes.append([x, y, width, height])
                else:
                    # Custom format: x y width height
                    print(f"Detected custom format in {label_file_path}")
                    x, y, width, height = map(int, nums)
                    bounding_boxes.append([x, y, width, height])
            
            else:
                print(f"Warning: Unrecognized format in {label_file_path}, line {line_num}: {line}")
                continue
                
    except Exception as e:
        print(f"Error parsing label file {label_file_path}: {e}")
        return []
    
    print(f"Extracted {len(bounding_boxes)} bounding box(es) from {label_file_path}")
    return bounding_boxes

def get_image_dimensions(image_path):
    """
    Get image dimensions using OpenCV (if available) or PIL
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            return width, height
    except ImportError:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # PIL returns (width, height)
        except ImportError:
            print("Warning: Neither OpenCV nor PIL available for reading image dimensions")
            print("YOLO format conversion may not work correctly")
            return None, None
    
    return None, None

def create_ground_truth(images_dir, labels_dir, output_file="output/ground_truth.json"):
    """
    Create ground truth JSON file from images and labels directories
    """
    ground_truth = {}
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not image_files:
        print(f"Error: No image files found in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} image file(s) in {images_dir}")
    
    for image_path in sorted(image_files):
        image_filename = os.path.basename(image_path)
        base_name = os.path.splitext(image_filename)[0]
        
        print(f"\nProcessing: {image_filename}")
        
        # Extract emotion from filename
        emotion = extract_emotion_from_filename(image_filename)
        
        # Look for corresponding label file
        label_file_path = os.path.join(labels_dir, base_name + '.txt')
        
        # Get image dimensions (needed for YOLO format)
        img_width, img_height = get_image_dimensions(image_path)
        if img_width:
            print(f"Image dimensions: {img_width}x{img_height}")
        
        # Parse bounding boxes from label file
        bounding_boxes = parse_label_file(label_file_path, img_width, img_height)
        
        # Create ground truth entry
        if emotion != "Unknown" or bounding_boxes:  # Include if we have emotion or bounding boxes
            # If we have multiple bounding boxes but only one emotion, replicate the emotion
            emotions = [emotion] * max(1, len(bounding_boxes))
            
            ground_truth[image_filename] = {
                "bounding_boxes": bounding_boxes,
                "emotions": emotions
            }
            
            print(f"Added to ground truth: {emotion} with {len(bounding_boxes)} bounding box(es)")
        else:
            print(f"Skipped: No emotion or bounding boxes found for {image_filename}")
    
    # Save ground truth JSON
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"\n✅ Ground truth saved to: {output_file}")
        print(f"📊 Total images processed: {len(ground_truth)}")
        
        # Print summary
        emotion_counts = {}
        total_faces = 0
        for data in ground_truth.values():
            total_faces += len(data['bounding_boxes'])
            for emotion in data['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"📊 Total faces: {total_faces}")
        print(f"📊 Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"   {emotion}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving ground truth file: {e}")
        return False

def print_sample_formats():
    """Print examples of supported label file formats"""
    print("\n📋 Supported label file formats:")
    print("=" * 40)
    
    print("\n1. YOLO Format (normalized coordinates):")
    print("   0 0.5 0.5 0.3 0.4")
    print("   Format: class_id center_x center_y width height")
    print("   Note: All values are normalized (0.0 to 1.0)")
    
    print("\n2. Pascal VOC Format (absolute coordinates):")
    print("   100 150 250 300")
    print("   Format: xmin ymin xmax ymax")
    
    print("\n3. Custom Format (absolute coordinates):")
    print("   100 150 150 150")
    print("   Format: x y width height")
    
    print("\n📝 Notes:")
    print("   - Empty lines and lines starting with # are ignored")
    print("   - Multiple faces per image: add multiple lines to the label file")
    print("   - If no label file exists, only emotion from filename will be used")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 create_ground_truth.py <images_dir> <labels_dir> [output_file]")
        print("\nExample:")
        print("   python3 create_ground_truth.py data/images data/labels")
        print("   python3 create_ground_truth.py data/images data/labels my_ground_truth.json")
        print("\nExpected filename format:")
        print("   happy (1).jpg, sad (2).jpg, angry (3).jpg, etc.")
        print_sample_formats()
        return
    
    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "output/ground_truth.json"
    
    # Validate directories
    if not os.path.exists(images_dir):
        print(f"❌ Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ Error: Labels directory not found: {labels_dir}")
        return
    
    print("🚀 Creating ground truth from your files...")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"📄 Output file: {output_file}")
    
    success = create_ground_truth(images_dir, labels_dir, output_file)
    
    if success:
        print(f"\n🎉 Success! Now you can run evaluation:")
        print(f"   python3 src/python/evaluation.py {images_dir} {output_file}")
    else:
        print(f"\n❌ Failed to create ground truth file")

if __name__ == "__main__":
    main()