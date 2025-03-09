# RDD: Respiratory Disease Detection

> 🏆 **Best Biomedical Hack** at **HackTJ 12.0**

**Low-cost screening for tuberculosis and other health warning signs using a microphone, a webcam and an Arduino.**
RDD listens to a cough through an Arduino microphone, turns it into a mel spectrogram and scores it with an
XGBoost model trained on real TB cough recordings. A webcam companion uses facial landmarks to aim the
microphone at the patient's mouth and to flag facial droop.

HackTJ 12.0 · March 2025

## 🩺 What it does

- **TB cough screening:** an Arduino with a MAX9814 microphone samples audio and streams it over serial. The
  Python app builds a mel spectrogram from each sample and the trained model returns a TB probability.
- **Mouth-tracking microphone:** dlib's 68-point face landmarks locate the mouth in the webcam feed, and two
  servos turn the microphone to follow it.
- **Facial droop detection:** the same landmarks measure eye and mouth asymmetry over time and raise an alert
  (on screen and to the Arduino) when droop persists.

## 🗂️ Files

| File | What it is |
|---|---|
| `FINAL_FEATURE_EXTRACTION.py` | Extracts 64-band mel spectrogram features from the CODA TB solicited cough recordings, with random Gaussian, pink and ambient noise augmentation |
| `FINAL_MODEL_NEW.py` | Loads the cough features with the clinical metadata, plus sensitivity, specificity, PPV/NPV, McNemar, Cohen's kappa and confidence-interval analysis |
| `FINALHACKTJ_NEW.py` | Live detector: reads audio from the Arduino, extracts features and scores them with the trained XGBoost model |
| `tb_detection.ino` | Arduino sketch: samples the MAX9814 microphone on `A0` and streams each window over serial at 115200 baud |
| `hacktj.py` | Webcam mouth tracking that sends servo angles to the Arduino |
| `bang.ino` | Arduino sketch: drives the two microphone servos (pins 9 and 10) from those angles |
| `hack2tj.py` | Webcam facial droop detection |
| `shape_predictor_68_face_landmarks.dat` | dlib's pretrained 68-point face landmark model |

## 🛠️ Built with

Python · XGBoost · scikit-learn · imbalanced-learn · librosa · TensorFlow · OpenCV · dlib · pySerial · Arduino

## ▶️ Running it

1. Install the Python dependencies: `numpy pandas librosa tensorflow xgboost scikit-learn imbalanced-learn
   matplotlib seaborn scipy joblib tqdm opencv-python dlib pyserial`.
2. Flash `tb_detection.ino` (cough screening) or `bang.ino` (mouth tracking) to the Arduino.
3. For training, place the CODA TB data under `data/raw/solicited_coughs/` (`audio/` and `metadata/`), then run
   `FINAL_FEATURE_EXTRACTION.py` followed by `FINAL_MODEL_NEW.py`.
4. Run `FINALHACKTJ_NEW.py` for live cough screening. It loads `tb_detection_model.joblib` (and `scaler.joblib` if
   present), which is not included in this repo.
5. Run `hacktj.py` or `hack2tj.py` for the webcam features. The serial port, camera index and model paths are
   set at the top of each script for the machine they were written on, so update them for yours.
