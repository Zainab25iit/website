import os
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import layers, models
import shutil

SR, DURATION, N_MELS = 16000, 2, 128
MODEL_PATH = './best_model.keras'

app = FastAPI(title="DeepVoiceAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def MBConv_Block(x, expansion, filters, stride):
    shortcut = x; in_ch = x.shape[-1]
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(expansion * in_ch, 1, padding='same', use_bias=False)(x)
    x = layers.ReLU(6.0)(x)
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and in_ch == filters: x = layers.Add()([shortcut, x])
    return x

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU(6.0)(x)
    
    x = MBConv_Block(x, 1, 16, 1)
    x = MBConv_Block(x, 6, 24, 2)
    x = MBConv_Block(x, 6, 40, 2)
    x = MBConv_Block(x, 6, 80, 2)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name='DeepfakeDetector_V3')
    return model

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Production model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] No model at {MODEL_PATH}. Initializing structural dummy.")
            model = build_model(input_shape=(128, 63, 1))
    except Exception as e:
        print(f"[CRITICAL] Error loading model: {e}")
        model = build_model(input_shape=(128, 63, 1))

def preprocess_audio(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=SR)
        target_length = SR * DURATION
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=512)
        db = librosa.power_to_db(mel, ref=np.max)
        
        db = np.nan_to_num(db, nan=-80.0, posinf=0.0, neginf=-80.0)
        db = np.clip(db, -80.0, 0.0)
        
        normalized_mel = (db + 40.0) / 20.0
        return normalized_mel[..., np.newaxis][np.newaxis, ...]
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

import uuid

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.mpeg', '.mpg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav, .mp3, and .mpeg are supported.")
    
    unique_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), unique_filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        features = preprocess_audio(temp_path)
        if features is None:
            raise HTTPException(status_code=500, detail="Error processing audio file.")
        
        prediction = model.predict(features)[0][0]
        
        label = "FAKE" if prediction > 0.5 else "REAL"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return {
            "filename": file.filename,
            "label": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": float(prediction)
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
