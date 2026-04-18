"""
Speech Emotion Recognition — Live Demo
=======================================
Usage:
    python emotion_demo.py --model best_crnn.pth --audio your_audio.wav

Requirements:
    pip install torch torchvision librosa soundfile matplotlib numpy scipy

Optional (for microphone recording):
    pip install sounddevice
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display

warnings.filterwarnings('ignore')

# ── Config (must match training config) ───────────────────────────────────────
SAMPLE_RATE   = 16000
DURATION      = 3.5
N_MELS        = 128
TARGET_LENGTH = 188
NUM_CLASSES   = 6
EMOTION_NAMES = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Emotion → (primary colour, text colour)
EMOTION_COLORS = {
    'Neutral'  : ('#8B8F9E', '#ffffff'),
    'Happy'    : ('#F5A623', '#ffffff'),
    'Sad'      : ('#4A90D9', '#ffffff'),
    'Angry'    : ('#D0021B', '#ffffff'),
    'Fearful'  : ('#7B68EE', '#ffffff'),
    'Disgust'  : ('#4CAF50', '#ffffff'),
    'Surprised': ('#FF6B6B', '#ffffff'),
}

EMOTION_EMOJI = {
    'Neutral': '😐', 'Happy': '😊', 'Sad': '😢',
    'Angry': '😠', 'Fearful': '😨', 'Disgust': '🤢', 'Surprised': '😲',
}

# ── Model Architecture (identical to training) ─────────────────────────────────

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
        mobilenet      = mobilenet_v2(weights=None)
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
        g, _ = self.gru(f)
        g   = self.drop_gru(g)
        ctx = self.attention(g)
        return self.classifier(ctx)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio(path: str):
    """Load a WAV/MP3/FLAC file, return (waveform, sr)."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def record_microphone(duration_sec=4):
    """Record from the default microphone. Requires sounddevice."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Install with: pip install sounddevice")
        sys.exit(1)
    print(f"\n🎙️  Recording {duration_sec}s... speak now!")
    audio = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Done recording.")
    return audio.flatten(), SAMPLE_RATE


def preprocess(y: np.ndarray) -> torch.Tensor:
    """Waveform → normalised 3-channel log-Mel tensor."""
    n_samples = int(SAMPLE_RATE * DURATION)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    mel     = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000, hop_length=256, n_fft=1024)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < TARGET_LENGTH:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_LENGTH - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TARGET_LENGTH]

    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    t = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return t.unsqueeze(0), log_mel           # (1,3,H,W), raw mel for plotting


def predict(model, tensor: torch.Tensor, device):
    """Run inference → (predicted_label, probabilities array)."""
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())
    return EMOTION_NAMES[pred], probs


# ── Visualisation ─────────────────────────────────────────────────────────────

def build_dashboard(y, sr, mel_2d, emotion, probs, audio_path):
    """
    Build a 4-panel dashboard:
      [0] waveform   [1] spectrogram
      [2] confidence bar chart   [3] result card
    """
    fig = plt.figure(figsize=(16, 9), facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.38,
        wspace=0.28,
        top=0.88, bottom=0.08,
        left=0.06, right=0.97
    )

    ax_wave  = fig.add_subplot(gs[0, 0])
    ax_mel   = fig.add_subplot(gs[0, 1])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_card  = fig.add_subplot(gs[1, 1])

    panel_bg  = '#1a1d27'
    text_main = '#e8e9ed'
    text_dim  = '#8a8fa8'
    accent    = EMOTION_COLORS[emotion][0]

    def style_ax(ax, title=''):
        ax.set_facecolor(panel_bg)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2e3248')
            sp.set_linewidth(0.8)
        ax.tick_params(colors=text_dim, labelsize=8)
        if title:
            ax.set_title(title, color=text_main, fontsize=10,
                         fontweight='bold', pad=8, loc='left')

    # ── 1. Waveform ───────────────────────────────────────────────────────────
    style_ax(ax_wave, 'Waveform')
    times = np.linspace(0, len(y) / sr, len(y))
    ax_wave.plot(times, y, color=accent, linewidth=0.6, alpha=0.9)
    ax_wave.axhline(0, color='#2e3248', linewidth=0.6)
    ax_wave.set_xlabel('Time (s)', color=text_dim, fontsize=8)
    ax_wave.set_ylabel('Amplitude', color=text_dim, fontsize=8)
    ax_wave.set_xlim(0, times[-1])

    # ── 2. Mel Spectrogram ────────────────────────────────────────────────────
    style_ax(ax_mel, 'Log-Mel Spectrogram')
    img = librosa.display.specshow(
        mel_2d,
        sr=SAMPLE_RATE,
        hop_length=256,
        x_axis='time',
        y_axis='mel',
        ax=ax_mel,
        cmap='magma'
    )
    cb = fig.colorbar(img, ax=ax_mel, format='%+2.0f dB', pad=0.02)
    cb.ax.yaxis.set_tick_params(color=text_dim, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=text_dim)
    cb.outline.set_edgecolor('#2e3248')
    ax_mel.set_xlabel('Time (s)', color=text_dim, fontsize=8)
    ax_mel.set_ylabel('Hz', color=text_dim, fontsize=8)

    # ── 3. Confidence Bar Chart ───────────────────────────────────────────────
    style_ax(ax_bar, 'Class Probabilities')
    sorted_idx = np.argsort(probs)[::-1]
    sorted_names = [EMOTION_NAMES[i] for i in sorted_idx]
    sorted_probs = [probs[i] * 100 for i in sorted_idx]
    bar_colors = [EMOTION_COLORS[n][0] for n in sorted_names]

    # Dim all bars except the top prediction
    bar_alphas = [1.0 if n == emotion else 0.45 for n in sorted_names]
    bars = ax_bar.barh(
        range(len(sorted_names)), sorted_probs,
        color=[c for c in bar_colors], height=0.6, zorder=3
    )
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)

    ax_bar.set_yticks(range(len(sorted_names)))
    ax_bar.set_yticklabels(
        [f"{EMOTION_EMOJI[n]}  {n}" for n in sorted_names],
        color=text_main, fontsize=9
    )
    ax_bar.set_xlabel('Confidence (%)', color=text_dim, fontsize=8)
    ax_bar.set_xlim(0, 105)
    ax_bar.invert_yaxis()
    ax_bar.grid(axis='x', color='#2e3248', linewidth=0.5, zorder=0)

    for bar, val in zip(bars, sorted_probs):
        ax_bar.text(
            val + 1.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', color=text_main, fontsize=8, fontweight='bold'
        )

    # ── 4. Result Card ────────────────────────────────────────────────────────
    ax_card.set_facecolor(panel_bg)
    for sp in ax_card.spines.values():
        sp.set_edgecolor(accent)
        sp.set_linewidth(1.5)
    ax_card.set_xticks([]); ax_card.set_yticks([])

    conf        = probs.max() * 100
    runner_up_i = np.argsort(probs)[-2]
    runner_up   = EMOTION_NAMES[runner_up_i]
    runner_conf = probs[runner_up_i] * 100

    # Accent band at top
    rect = FancyBboxPatch((0, 0.82), 1, 0.18,
                          boxstyle="square,pad=0",
                          transform=ax_card.transAxes,
                          color=accent, zorder=2, clip_on=False)
    ax_card.add_patch(rect)

    ax_card.text(0.5, 0.91, 'Predicted Emotion',
                 transform=ax_card.transAxes, ha='center', va='center',
                 color='white', fontsize=9, fontweight='bold', zorder=3)

    ax_card.text(0.5, 0.60,
                 f"{EMOTION_EMOJI[emotion]}  {emotion}",
                 transform=ax_card.transAxes, ha='center', va='center',
                 color=text_main, fontsize=28, fontweight='bold')

    ax_card.text(0.5, 0.42, f"Confidence: {conf:.1f}%",
                 transform=ax_card.transAxes, ha='center', va='center',
                 color=accent, fontsize=14, fontweight='bold')

    # Confidence gauge bar
    gauge_x, gauge_y, gauge_w, gauge_h = 0.1, 0.32, 0.8, 0.045
    ax_card.add_patch(FancyBboxPatch(
        (gauge_x, gauge_y), gauge_w, gauge_h,
        boxstyle="round,pad=0.01", transform=ax_card.transAxes,
        color='#2e3248', zorder=2, clip_on=False))
    ax_card.add_patch(FancyBboxPatch(
        (gauge_x, gauge_y), gauge_w * conf / 100, gauge_h,
        boxstyle="round,pad=0.01", transform=ax_card.transAxes,
        color=accent, zorder=3, clip_on=False))

    ax_card.text(0.5, 0.22,
                 f"Runner-up: {EMOTION_EMOJI[runner_up]} {runner_up} ({runner_conf:.1f}%)",
                 transform=ax_card.transAxes, ha='center', va='center',
                 color=text_dim, fontsize=9)

    src_label = os.path.basename(audio_path) if audio_path else 'microphone'
    ax_card.text(0.5, 0.10, f"Source: {src_label}",
                 transform=ax_card.transAxes, ha='center', va='center',
                 color=text_dim, fontsize=8)

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.95,
             'Speech Emotion Recognition — Live Demo',
             ha='center', color=text_main, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.91,
             'MobileNetV2 + BiGRU + Temporal Attention  |  RAVDESS · CREMA-D · TESS · SAVEE',
             ha='center', color=text_dim, fontsize=8)

    plt.savefig('emotion_result.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("\n📊 Dashboard saved → emotion_result.png")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Speech Emotion Recognition — Live Demo')
    parser.add_argument('--model', default='best_crnn.pth',
                        help='Path to .pth checkpoint (default: best_crnn.pth)')
    parser.add_argument('--audio', default=None,
                        help='Path to audio file (WAV/MP3/FLAC). '
                             'Omit to record from microphone.')
    parser.add_argument('--record_sec', type=int, default=4,
                        help='Seconds to record if using mic (default: 4)')
    parser.add_argument('--classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes (default: 7)')
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")

    model = LightweightCRNN(num_classes=args.classes).to(device)
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded {args.model}  ({n_params/1e6:.1f}M parameters)")

    # ── Load or record audio ───────────────────────────────────────────────────
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"❌ Audio file not found: {args.audio}")
            sys.exit(1)
        print(f"\n🎵 Loading: {args.audio}")
        y, sr = load_audio(args.audio)
    else:
        y, sr = record_microphone(args.record_sec)

    duration_loaded = len(y) / sr
    print(f"   Duration: {duration_loaded:.2f}s  |  Sample rate: {sr} Hz")

    # ── Preprocess & predict ───────────────────────────────────────────────────
    tensor, mel_2d = preprocess(y)
    emotion, probs = predict(model, tensor, device)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  🎯 Predicted: {EMOTION_EMOJI[emotion]}  {emotion}")
    print(f"  📊 Confidence: {probs.max()*100:.1f}%")
    print("=" * 50)
    print("\nAll class probabilities:")
    for i, (name, prob) in enumerate(zip(EMOTION_NAMES, probs)):
        bar = '█' * int(prob * 30)
        marker = ' ← predicted' if name == emotion else ''
        print(f"  {name:<10} {bar:<30} {prob*100:5.1f}%{marker}")

    # ── Build visual dashboard ─────────────────────────────────────────────────
    print("\n🖼️  Generating dashboard...")
    build_dashboard(y, sr, mel_2d, emotion, probs,
                    audio_path=args.audio or 'microphone')


if __name__ == '__main__':
    main()
