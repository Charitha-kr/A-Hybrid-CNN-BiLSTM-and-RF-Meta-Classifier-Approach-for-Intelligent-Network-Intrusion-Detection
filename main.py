import os
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Try importing TensorFlow
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed. DL model will be skipped.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STREAM_FILE = os.path.join(BASE_DIR, 'data', 'stream.csv')

# --- GLOBAL VARIABLES ---
server_stats = {
    "total_scanned": 0,
    "malicious_count": 0,
    "current_status": "Secure"
}

attack_history_log = [] 

# NEW: Sliding Window List (Stores 0 for Benign, 1 for Attack)
# We will keep only the last 100 packets here
recent_window = [] 

rf_binary = None
cnn_multiclass = None
scaler = None
label_encoder = None

# --- LOAD MODELS ---
print("--------- LOADING SYSTEM ---------")
try:
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "random_forest_binary.pkl"), "rb") as f:
        rf_binary = pickle.load(f)
    if TF_AVAILABLE:
        cnn_multiclass = load_model(os.path.join(MODELS_DIR, "cnn_bilstm_multiclass.h5"))
    print("✅ System Ready")
except Exception as e:
    print(f"❌ Error: {e}")

# --- PREDICTION LOGIC ---
def predict_packet_hybrid(raw_features):
    if not scaler or not rf_binary: return "Initializing...", 0.0
    try:
        features_2d = raw_features.reshape(1, -1)
        X_scaled = scaler.transform(features_2d)
        rf_pred = rf_binary.predict(X_scaled)[0] 

        if rf_pred == 0: return "BENIGN", 99.9
        else:
            if cnn_multiclass and label_encoder:
                X_seq = np.expand_dims(X_scaled, -1)
                cnn_probs = cnn_multiclass.predict(X_seq, verbose=0)
                class_idx = np.argmax(cnn_probs, axis=1)[0]
                confidence = float(np.max(cnn_probs) * 100)
                return label_encoder.inverse_transform([class_idx])[0], confidence
            return "Malicious (General)", 80.0
    except: return "Error", 0.0

# --- API ENDPOINTS ---

@app.get("/api/status")
def get_status():
    return {"status": "Online (Cascade AI)" if rf_binary else "Offline"}

@app.get("/api/stats")
def get_stats():
    # NEW: Calculate System Health based on Sliding Window
    global recent_window
    
    health_score = 100
    
    if len(recent_window) > 0:
        # Count how many attacks are in the last 100 packets
        recent_attacks = sum(recent_window)
        window_size = len(recent_window)
        
        # Calculate Attack Ratio (0.0 to 1.0)
        attack_ratio = recent_attacks / window_size
        
        # Calculate Health: 100% minus the attack percentage
        # Example: 20 attacks in 100 packets = 0.2 ratio = 80% Health
        health_score = max(0, 100 - int(attack_ratio * 100))

    return {
        "total_packets": server_stats["total_scanned"],
        "malicious_detected": server_stats["malicious_count"],
        "active_threats": 1 if server_stats["current_status"] == "Under Attack" else 0,
        "system_health": f"{health_score}%"  # Returns e.g. "85%"
    }

@app.get("/api/attack-history")
def get_history():
    return attack_history_log

@app.get("/api/live-traffic")
def get_live_traffic():
    global server_stats, attack_history_log, recent_window

    row_data = None
    if os.path.exists(STREAM_FILE):
        try:
            df = pd.read_csv(STREAM_FILE)
            if not df.empty:
                last_line = df.iloc[-1]
                row_data = (last_line.to_dict(), last_line.values.astype(float))
        except: pass

    if not row_data: return {"error": "Waiting..."}
    row_dict, features = row_data

    prediction, confidence = predict_packet_hybrid(features)

    server_stats["total_scanned"] += 1
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")

    # NEW: Update Sliding Window
    # Append 1 if attack, 0 if benign
    is_attack = 1 if prediction != "BENIGN" else 0
    recent_window.append(is_attack)
    
    # Keep only the last 100 entries
    if len(recent_window) > 100:
        recent_window.pop(0)

    # If Attack, Save to History Log
    if prediction != "BENIGN":
        server_stats["malicious_count"] += 1
        server_stats["current_status"] = "Under Attack"
        
        new_record = {
            "id": server_stats["total_scanned"],
            "timestamp": timestamp,
            "type": prediction,
            "confidence": round(confidence, 2),
            "details": f"Duration: {row_dict.get('Flow Duration', 0)}"
        }
        attack_history_log.insert(0, new_record) 
        attack_history_log = attack_history_log[:100] 
        
    else:
        server_stats["current_status"] = "Secure"

    return {
        "timestamp": timestamp,
        "flow_duration": float(row_dict.get('Flow Duration', 0)),
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }