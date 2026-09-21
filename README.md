# Computer Vision – Final Project: Face Detection & Emotion Recognition

This project implements an end-to-end **face detection + emotion recognition** system for the Computer Vision course final assignment. It combines a classical C++/OpenCV detector with a Python deep-learning emotion classifier, glued together by a C++ orchestrator.

The pipeline is:

1. Detect faces in an input image with the **Viola–Jones** algorithm (`cv::CascadeClassifier`, Haar cascade).
2. Crop and save each detected face region.
3. Run **emotion recognition** on each face crop via a Python script (built on **DeepFace**).
4. Merge detections + emotion labels into a single annotated output image and a results file.
5. Evaluate predictions against ground-truth labels (confusion matrix, classification report).

---

## Author

* [@adelinapopa02](https://github.com/adelinapopa02)

---

## Project overview

| Module | Role |
|---|---|
| `FaceDetector` (C++) | Wraps the Haar cascade classifier; detects, filters and saves face regions. |
| `Integration` (C++) | Orchestrates the pipeline: runs face detection, calls the Python emotion script as a subprocess, merges results into the final annotated image. |
| `emotion_recognition.py` | Loads the face crops and predicts an emotion label for each, using DeepFace. |
| `evaluation.py` | Compares the pipeline's output against ground-truth labels and reports accuracy metrics. |
| `main.cpp` | CLI entry point exposing single-image and batch modes. |

### Modes (`face_emotion_system`)

```
--detect-only <image>          Face detection only
--integrate <image>            Face detection + emotion recognition on a single image
--batch-detect <dir>           Face detection on every image in a directory
--batch-integrate <dir>        Full pipeline on every image in a directory
```

Options: `--cascade <path>`, `--script <path>`, `--extensions jpg,png,...`.

---

## Setup

Python side (emotion recognition needs TensorFlow/DeepFace):

```bash
./setup.sh                 # creates cv_venv/ and installs numpy, opencv-python, tensorflow, tf-keras, deepface, matplotlib, seaborn, scikit-learn
source cv_venv/bin/activate
```

C++ side:

```bash
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
./face_emotion_system --integrate ../data/images/happy_1.jpg
./face_emotion_system --batch-integrate ../data/images/
python3 ../src/python/evaluation.py ../output ../data/labels
```

---

## Known issue

`main.cpp`, `Integration.h`/`.cpp`, `FaceDetector.h`, `CMakeLists.txt` and `emotion_recognition.py` still contain **unresolved `<<<<<<< HEAD` / `>>>>>>> deepface-version` git conflict markers** from a past merge — the code as committed will not compile/run as-is until that merge is finalized. Worth cleaning up before relying on this README's build steps.
