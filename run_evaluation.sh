#!/bin/bash

# Output-based evaluation script for Face Detection and Emotion Recognition System
# This script evaluates only the images that have been processed (exist in output/)

echo "Face Detection and Emotion Recognition - Output-based Evaluation"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the project root directory (where CMakeLists.txt is located)"
    exit 1
fi

# Check if output directory exists and has results
if [ ! -d "output" ]; then
    echo "Error: output directory not found"
    echo "   Please run your face detection system first to generate results"
    exit 1
fi

# Check if there are any emotion results in output
emotion_results=$(ls output/*_emotion_results.json 2>/dev/null | wc -l)
if [ $emotion_results -eq 0 ]; then
    echo "Error: No emotion results found in output/"
    echo "   Please run the integrated system on some images first:"
    echo "   cd build && ./face_emotion_system --integrate ../data/images/your_image.jpg"
    exit 1
fi

echo "Found $emotion_results processed image(s) in output/"

# Check if data directories exist (needed for ground truth creation)
if [ ! -d "data/images" ]; then
    echo "Error: data/images directory not found"
    exit 1
fi

if [ ! -d "data/labels" ]; then
    echo "Warning: data/labels directory not found"
    echo "   Ground truth will be created from filenames only (no bounding boxes)"
    mkdir -p data/labels  # Create empty labels directory
fi

echo "Starting output-based evaluation..."

# Step 1: Extract image names from existing results
echo ""
echo "Step 1: Identifying processed images from output..."
echo "=================================================="

processed_images=()
for result_file in output/*_emotion_results.json; do
    if [ -f "$result_file" ]; then
        # Extract base name from result file
        # e.g., "output/happy_1_emotion_results.json" -> "happy_1"
        base_name=$(basename "$result_file" _emotion_results.json)
        
        # Look for corresponding original image
        found_image=""
        for ext in jpg jpeg png bmp; do
            # Try different possible original names
            for original in "data/images/${base_name}.${ext}" "data/images/${base_name} (1).${ext}" "data/images/${base_name}(1).${ext}"; do
                if [ -f "$original" ]; then
                    found_image="$original"
                    break 2
                fi
            done
        done
        
        if [ -n "$found_image" ]; then
            processed_images+=("$(basename "$found_image")")
            echo "   Found processed: $(basename "$found_image")"
        else
            echo "   Warning: Could not find original image for $base_name"
        fi
    fi
done

if [ ${#processed_images[@]} -eq 0 ]; then
    echo "Error: Could not match any result files to original images"
    exit 1
fi

echo "Total processed images to evaluate: ${#processed_images[@]}"

# Step 2: Create ground truth for only the processed images
echo ""
echo "Step 2: Creating ground truth for processed images only..."
echo "========================================================="

# Create a temporary directory with only the processed images
temp_images_dir="temp_eval_images"
mkdir -p "$temp_images_dir"

# Copy only the processed images to temp directory
for img in "${processed_images[@]}"; do
    if [ -f "data/images/$img" ]; then
        cp "data/images/$img" "$temp_images_dir/"
        echo "   Prepared: $img"
    fi
done

# Create ground truth from only the processed images
echo ""
echo "Creating ground truth from processed images..."
if python3 src/python/ground_truth.py "$temp_images_dir" data/labels output/ground_truth.json; then
    echo "Ground truth created successfully for ${#processed_images[@]} processed images!"
else
    echo "Failed to create ground truth. Check the error messages above."
    rm -rf "$temp_images_dir"
    exit 1
fi

# Clean up temporary directory
rm -rf "$temp_images_dir"

# Step 3: Run evaluation on the processed results
echo ""
echo "Step 3: Running evaluation on processed results..."
echo "=================================================="

if python3 src/python/evaluation.py output output/ground_truth.json; then
    echo ""
    echo "Evaluation completed successfully!"
    echo ""
    echo "Generated files:"
    echo "   - evaluation_report.txt (detailed performance metrics)"
    echo "   - confusion_matrix.png (emotion classification matrix)"
    echo "   - output/ground_truth.json (ground truth for processed images)"
    echo ""
    echo "Evaluation Summary:"
    echo "   Images processed: ${#processed_images[@]}"
    echo "   Images evaluated: ${#processed_images[@]}"
    echo ""
    echo "To view results:"
    echo "   cat evaluation_report.txt"
    echo "   open confusion_matrix.png  # (on macOS)"
    echo ""
    echo "Processed images:"
    for img in "${processed_images[@]}"; do
        echo "   - $img"
    done
    
else
    echo "Evaluation failed. Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "   - Missing Python dependencies (tensorflow, opencv-python, etc.)"
    echo "   - Incorrect label file format in data/labels/"
    echo "   - Mismatched result files in output/"
    exit 1
fi

echo ""
echo "Output-based evaluation complete!"
echo "Only the images you actually processed were evaluated."