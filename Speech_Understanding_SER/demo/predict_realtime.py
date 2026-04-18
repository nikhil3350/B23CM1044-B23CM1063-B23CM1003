"""
Real-Time Speech Emotion Recognition
Uses best_crnn.pth weights + microphone + webcam overlay
Run: python predict_realtime.py
Press Q to quit.
"""

import os, time, threading, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import sounddevice as sd
import cv2

# ─────────────────────────────────────────────────────────────────
# CONFIG — must match your training config exactly
# ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
DURATION       = 3.5          # seconds of audio per inference
N_MELS         = 128
TARGET_LENGTH  = 188
NUM_CLASSES    = 6
WEIGHTS_PATH   = "best_crnn.pth"   # path to your downloaded .pth

EMOTION_NAMES  = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust']

# Emoji + color (BGR) for each emotion — for the webcam overlay
EMOTION_STYLE  = {
    'Neutral' : ('😐', (200, 200, 200)),
    'Happy'   : ('😄', (0,   220, 100)),
    'Sad'     : ('😢', (200, 100,  50)),
    'Angry'   : ('😠', (0,    0,  230)),
    'Fearful' : ('😨', (180,  50, 180)),
    'Disgust' : ('🤢', (0,   180,  60)),
}

device = torch.device("cpu")   # change to "cuda" if you have a GPU


# ─────────────────────────────────────────────────────────────────
# MODEL — copy-pasted exactly from your notebook Cell 5
# ─────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)


class LightweightCRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, gru_hidden=192, dropout=0.45):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        mobilenet = mobilenet_v2(weights=None)          # no download needed at inference
        self.features  = mobilenet.features
        self.ln_in     = nn.LayerNorm(1280)
        self.gru       = nn.GRU(1280, gru_hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
        self.drop_gru  = nn.Dropout(dropout)
        self.attention = TemporalAttention(gru_hidden * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 256), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f   = self.features(x)
        f   = f.mean(dim=2).permute(0, 2, 1)
        f   = self.ln_in(f)
        g, _= self.gru(f)
        g   = self.drop_gru(g)
        ctx = self.attention(g)
        return self.classifier(ctx)


# ─────────────────────────────────────────────────────────────────
# LOAD WEIGHTS
# ─────────────────────────────────────────────────────────────────
def load_model(path):
    model = LightweightCRNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded weights from: {path}")
    return model


# ─────────────────────────────────────────────────────────────────
# AUDIO → MEL SPECTROGRAM (same as training pipeline)
# ─────────────────────────────────────────────────────────────────
def extract_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, target_length=TARGET_LENGTH):
    samples = int(sr * DURATION)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    else:
        y = y[:samples]

    mel     = librosa.feature.melspectrogram(
                  y=y, sr=sr, n_mels=n_mels, fmax=8000,
                  hop_length=256, n_fft=1024)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < target_length:
        log_mel = np.pad(log_mel,
                         ((0,0),(0, target_length - log_mel.shape[1])),
                         mode='constant')
    else:
        log_mel = log_mel[:, :target_length]

    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    # 3-channel tensor for MobileNetV2
    t = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
    return t.unsqueeze(0)   # (1, 3, 128, 188)


# ─────────────────────────────────────────────────────────────────
# SHARED STATE between audio thread and display thread
# ─────────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.audio_buffer   = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
        self.emotion        = "Listening..."
        self.probabilities  = np.ones(NUM_CLASSES) / NUM_CLASSES   # uniform start
        self.confidence     = 0.0
        self.last_inference = 0.0
        self.history        = collections.deque(maxlen=5)    # smooth over 5 preds


state = SharedState()


# ─────────────────────────────────────────────────────────────────
# AUDIO CALLBACK — called by sounddevice in a background thread
# ─────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    with state.lock:
        state.audio_buffer = np.roll(state.audio_buffer, -len(mono))
        state.audio_buffer[-len(mono):] = mono


# ─────────────────────────────────────────────────────────────────
# INFERENCE THREAD — runs every 1.5 seconds (sliding window)
# ─────────────────────────────────────────────────────────────────
def inference_loop(model, interval=1.5):
    while True:
        time.sleep(interval)
        with state.lock:
            audio_snap = state.audio_buffer.copy()

        try:
            tensor = extract_mel(audio_snap)
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            conf     = float(probs[pred_idx])

            # Temporal smoothing: vote over last 5 predictions
            state.history.append(pred_idx)
            smoothed_idx = collections.Counter(state.history).most_common(1)[0][0]

            with state.lock:
                state.emotion       = EMOTION_NAMES[smoothed_idx]
                state.probabilities = probs
                state.confidence    = conf
                state.last_inference= time.time()

        except Exception as e:
            print(f"Inference error: {e}")


# ─────────────────────────────────────────────────────────────────
# WEBCAM OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────────
def draw_overlay(frame, emotion, probs, confidence):
    h, w = frame.shape[:2]
    emoji, color = EMOTION_STYLE.get(emotion, ('❓', (255,255,255)))

    # Semi-transparent dark panel on the left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "EMOTION DETECTOR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.putText(frame, "─" * 28, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    # Current prediction (large)
    cv2.putText(frame, emotion, (10, 90),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2)
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Probability bars for all 6 emotions
    bar_y = 150
    for i, (emo, prob) in enumerate(zip(EMOTION_NAMES, probs)):
        _, bar_color = EMOTION_STYLE[emo]
        bar_w = int(prob * 240)
        label_color = bar_color if emo == emotion else (130,130,130)

        cv2.putText(frame, f"{emo:<9}", (10, bar_y + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
        cv2.rectangle(frame, (95, bar_y + i*35 - 12),
                              (95 + bar_w, bar_y + i*35 + 3),
                              bar_color, -1)
        cv2.putText(frame, f"{prob*100:.1f}%", (98 + bar_w, bar_y + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    # Recording indicator (pulsing red dot)
    pulse = int(time.time() * 2) % 2
    dot_color = (0, 0, 220) if pulse else (0, 0, 120)
    cv2.circle(frame, (260, 25), 8, dot_color, -1)
    cv2.putText(frame, "REC", (225, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    # Bottom hint
    cv2.putText(frame, "Press Q to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    return frame


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # Check weights file exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Weights not found at: {WEIGHTS_PATH}")
        print("   Make sure best_crnn.pth is in the same folder as this script.")
        return

    print("🔄 Loading model...")
    model = load_model(WEIGHTS_PATH)

    print("🎤 Starting microphone stream...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=1024,
        callback=audio_callback
    )

    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Running audio-only mode.")
        cap = None

    # Start inference in background thread
    inf_thread = threading.Thread(target=inference_loop, args=(model,), daemon=True)
    inf_thread.start()
    print("✅ Real-time prediction running! Press Q in the webcam window to quit.\n")

    with stream:
        while True:
            with state.lock:
                emotion    = state.emotion
                probs      = state.probabilities.copy()
                confidence = state.confidence

            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)   # mirror
                    frame = draw_overlay(frame, emotion, probs, confidence)
                    cv2.imshow("Speech Emotion Recognition — Press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # No webcam — just print to terminal
                bar = '█' * int(confidence * 20)
                print(f"\r🎭 {emotion:<10} [{bar:<20}] {confidence*100:.1f}%  ", end='', flush=True)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Stopped.")


if __name__ == "__main__":
    main()