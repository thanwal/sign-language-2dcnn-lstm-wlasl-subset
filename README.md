# Sign Language Recognition with 2D CNN + LSTM

A deep learning pipeline to recognize American Sign Language (ASL) signs from videos using a compact 2D CNN + LSTM model trained on a mini version of the WLASL dataset.

## 🚀 Project Overview
This project implements a lightweight and modular ASL recognition system that:
- Extracts and preprocesses videos from the WLASL dataset
- Selects meaningful frames
- Extracts hand landmarks using MediaPipe
- Trains a 2D CNN + LSTM model on sequences of landmarks
- Runs real-time inference on webcam input

Due to limited GPU resources, a mini-dataset was created and used instead of the full WLASL dataset.

## 📁 Repository Structure
```
├── class_list.txt              # List of selected ASL sign classes
├── move_video.py               # Moves videos matching classes from WLASL metadata
├── sort_videos.py              # Organizes videos into class folders
├── frame_extract.py            # Extracts 15 frames per video with visible hands
├── landmark_extract.py         # Extracts 3D hand landmarks using MediaPipe
├── data_split.py               # Splits data into train, val, test sets
├── final_model.py              # 2D CNN + LSTM training script
├── model.h5                    # Trained model file
├── test_model.py               # Real-time sign recognition via webcam
└── README.md                   # Project documentation (this file)
```

## 📦 Dataset
- **Original WLASL Dataset:** [WLASL Processed on Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- **Mini Dataset (Subset for this Project):** [Mini Dataset Link](http://www.something) *(to be updated)*

## ⚙️ Installation
```bash
# Clone this repo
$ git clone https://github.com/mrthanwal/sign-language-2dcnn-lstm-wlasl-subset.git
$ cd sign-language-2dcnn-lstm-wlasl-subset

# Create virtual environment (optional)
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

```

> ⚠️ Ensure you have MediaPipe, TensorFlow, OpenCV, and numpy installed.

## 🧠 Model Architecture
- **2D CNN**: Extracts features from each hand landmark frame (15 total)
- **LSTM**: Learns temporal sequences of sign motion
- **Input shape**: (15, 126, 1) where 126 is the MediaPipe hand landmark vector (2 hands × 21 points × 3 coords)

## 🛠 Usage

### Step 1: Prepare Mini Dataset
```bash
python move_video.py
python sort_videos.py
```

### Step 2: Preprocess Videos
```bash
python frame_extract.py
python landmark_extract.py
python data_split.py
```

### Step 3: Train the Model
```bash
python final_model.py
```

### Step 4: Test Model in Real-Time
```bash
python test_model.py
```

## ✅ Results
- Lightweight model suitable for low-resource environments
- Accurate real-time sign recognition with webcam input
- Better performance than 3D CNN baseline in limited compute conditions

## 🔮 Future Enhancements
- Convert to TensorFlow Lite for edge deployment
- Expand dataset with more classes and samples
- Use Transformer-based models for improved temporal learning
- Integrate a user-friendly GUI or TTS for accessibility

## 📄 License
This project is open-source.
## 🙌 Acknowledgements
- WLASL Dataset authors 
- MediaPipe by Google for hand landmark extraction

---

**Author:** [@thanwal](https://github.com/thanwal)
