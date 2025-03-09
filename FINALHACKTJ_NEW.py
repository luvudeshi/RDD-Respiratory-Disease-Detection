import serial
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import os
import random
import xgboost as xgb
from scipy import signal
import serial.tools.list_ports
import threading
import queue
import warnings
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import traceback

audio_data = []
mel_spec_data = np.zeros((128, 64))
prediction_history = []
time_history = []
audio_stats = []

positive_count = 0
negative_count = 0
total_count = 0

is_running = False
is_processing = False
is_connected = False
waiting_for_arduino_data = False
data_requested = False

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
FMAX = 8000

data_queue = queue.Queue()

root = None
canvas = None
start_button = None
stop_button = None
status_label = None
result_text = None
stats_label = None
progress_bar = None
connection_indicator = None

model = None
scaler = None
ser = None
arduino_thread = None
stop_event = None

def load_model():
    try:
        model = joblib.load("tb_detection_model.joblib")
        try:
            scaler = joblib.load("scaler.joblib")
        except:
            scaler = None
        return model, scaler
    except:
        if os.path.exists("test_model.joblib"):
            try:
                model = joblib.load("test_model.joblib")
                return model, None
            except:
                pass
    return None, None

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description or "FTDI" in port.description:
            return port.device
    return 'COM3'

def connect_to_arduino():
    global is_connected
    port = find_arduino_port()
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        is_connected = True
        return ser
    except:
        is_connected = False
        return None

def arduino_reader_thread(ser, stop_event):
    global is_connected, waiting_for_arduino_data, data_requested
    receiving_data = False
    buffer = []
    if ser and ser.is_open:
        ser.reset_input_buffer()
    while not stop_event.is_set():
        try:
            if ser and ser.is_open and ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "START_DATA" in line:
                    receiving_data = True
                    buffer = []
                elif "END_DATA" in line:
                    receiving_data = False
                    if buffer and data_requested:
                        data_queue.put(buffer.copy())
                        data_requested = False
                elif receiving_data:
                    buffer.append(line)
            else:
                time.sleep(0.01)
        except:
            is_connected = False
            time.sleep(0.5)

def extract_features(audio_data, sr=SAMPLE_RATE):
    audio = np.array(audio_data, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec = mel_spec.T
    if mel_spec.shape[1] != N_MELS:
        mel_spec = mel_spec[:, :N_MELS]
    if mel_spec.shape[0] < 128:
        padding = 128 - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, padding), (0, 0)), mode='constant')
    elif mel_spec.shape[0] > 128:
        mel_spec = mel_spec[:128, :]
    return mel_spec

def create_feature_vector(mel_spec):
    features_flat = mel_spec.flatten()
    clinical_features = np.array([random.uniform(20, 70)/100, random.uniform(1.5, 1.9), random.uniform(45, 85),
                                  random.uniform(5, 30), random.uniform(70, 100), random.uniform(36.5, 39),
                                  random.uniform(0.2, 0.8)])
    return np.hstack((features_flat, clinical_features)).reshape(1, -1)

def predict_tb(model, scaler, features):
    global positive_count, negative_count, total_count
    if model is None:
        prediction = random.random()
    else:
        features_scaled = scaler.transform(features) if scaler is not None else features
        prediction = model.predict_proba(features_scaled)[0, 1]
    is_positive = prediction > 0.3
    if is_positive:
        positive_count += 1
    else:
        negative_count += 1
    total_count += 1
    return prediction


global model, scaler, ser, arduino_thread, stop_event
model, scaler = load_model()
try:
    ser = connect_to_arduino()
except:
    ser = None
stop_event = threading.Event()
if ser and ser.is_open:
    try:
        arduino_thread = threading.Thread(target=arduino_reader_thread, args=(ser, stop_event))
        arduino_thread.daemon = True
        arduino_thread.start()
    except:
        ser = None
if ser and hasattr(ser, 'is_open') and ser.is_open:
    ser.close()
print(f"Positive detections: {positive_count}")
print(f"Negative detections: {negative_count}")
print(f"Total samples: {total_count}")
if total_count > 0:
    print(f"Positive rate: {positive_count/total_count*100:.1f}%")
