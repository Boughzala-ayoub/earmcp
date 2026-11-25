"""earmcp: an MCP server that lets an AI "hear" local audio files.

Setup:   python -m venv .venv --system-site-packages && .venv\\Scripts\\pip install -r requirements.txt
Warm up: .venv\\Scripts\\python selftest.py   (downloads the models once, then checks every tool)
Claude:  claude mcp add earmcp --scope user -- <abs path>\\.venv\\Scripts\\python.exe <abs path>\\server.py

Models are swappable via env vars EARMCP_WHISPER / EARMCP_CLASSIFIER (any Hugging Face id).
"""

import io
import os
from functools import cache

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mcp.server.mcpserver import Image, MCPServer  # noqa: E402

SR = 16000  # both models expect 16 kHz mono
WHISPER = os.environ.get("EARMCP_WHISPER", "openai/whisper-base")
CLASSIFIER = os.environ.get("EARMCP_CLASSIFIER", "MIT/ast-finetuned-audioset-10-10-0.4593")
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

mcp = MCPServer(
    "earmcp",
    instructions=(
        "Tools for understanding local audio files (wav, flac, ogg, mp3). Pass absolute paths. "
        "Start with classify_sounds to learn what is in the audio, then transcribe for speech, "
        "analyze_audio for levels/pitch/tempo/key, and spectrogram to look at it. "
        "Every tool takes start/duration (seconds) to focus on part of a long file."
    ),
)


def load(path: str, sr: int | None = SR, start: float = 0.0, duration: float | None = None, mono: bool = True):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ValueError(f"No such file: {path}")
    try:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=start, duration=duration)
    except Exception as e:
        raise ValueError(f"Can't decode {path}: {e}. wav/flac/ogg/mp3 work; m4a/aac/video need ffmpeg on PATH.") from e
    if y.size == 0:
        raise ValueError(f"No audio in {path} at start={start}s")
    return y, sr


def r(x, n=2):
    return None if x is None else round(float(x), n)


@mcp.tool()
def classify_sounds(path: str, start: float = 0.0, duration: float | None = None, top_k: int = 8) -> dict:
    """Identify what is audible (speech, music, instruments, animals, vehicles, alarms... 527 AudioSet
    classes) with a timeline of what happens in each 10-second window."""
    import torch  # ponytail: heavy imports stay lazy so the server starts fast

    y, _ = load(path, start=start, duration=duration)
    fe, model = _classifier()
    win = 10 * SR
    windows = [y[i : i + win] for i in range(0, len(y), win)]
    windows = [w for w in windows if len(w) >= SR] or windows[:1]  # drop a sub-second tail
    probs = []
    with torch.no_grad():
        for i in range(0, len(windows), 8):
            x = fe(windows[i : i + 8], sampling_rate=SR, return_tensors="pt")
            probs.append(torch.sigmoid(model(**x).logits))  # AudioSet is multi-label: sigmoid, not softmax
    p = torch.cat(probs).numpy()
    label = model.config.id2label
    overall = p.max(0)  # max over windows = "present somewhere"
    return {
        "top_sounds": [{"label": label[i], "confidence": r(overall[i])} for i in overall.argsort()[::-1][:top_k]],
        "timeline": [
            {
                "start_s": r(start + n * 10, 1),
                "end_s": r(start + n * 10 + len(windows[n]) / SR, 1),
                "sounds": [f"{label[i]} ({row[i]:.2f})" for i in row.argsort()[::-1][:3] if row[i] >= 0.15],
            }
            for n, row in enumerate(p)
        ],
    }


@mcp.tool()
def transcribe(path: str, start: float = 0.0, duration: float | None = None, language: str | None = None) -> dict:
    """Speech to text (Whisper, ~100 languages, auto-detected) with timestamped segments.
    language: optional, e.g. "english", "french", "arabic"."""
    y, _ = load(path, start=start, duration=duration)
    out = _asr()(
        {"raw": y, "sampling_rate": SR},
        return_timestamps=True,
        generate_kwargs={"language": language, "task": "transcribe"} if language else {"task": "transcribe"},
    )
    return {
        "text": out["text"].strip(),
        "segments": [
            {"start_s": r(start + (c["timestamp"][0] or 0), 1),
             "end_s": None if c["timestamp"][1] is None else r(start + c["timestamp"][1], 1),
             "text": c["text"].strip()}
            for c in out.get("chunks", [])
        ],
    }


@mcp.tool()
def analyze_audio(path: str, start: float = 0.0, duration: float | None = None) -> dict:
    """Measure the signal: duration, channels, loudness, clipping, silence and where the sound is,
    dominant frequency, brightness, pitch (melody/voice), and tempo + musical key (music only)."""
    raw, native_sr = load(path, sr=None, start=start, duration=duration, mono=False)
    channels = 1 if raw.ndim == 1 else raw.shape[0]
    sr = 22050
    y = librosa.resample(librosa.to_mono(raw), orig_sr=native_sr, target_sr=sr)

    peak = np.abs(raw).max()
    rms = np.sqrt(np.mean(y**2))
    db = lambda v: r(20 * np.log10(max(v, 1e-10)), 1)  # noqa: E731
    sounding = librosa.effects.split(y, top_db=40)
    spectrum = np.abs(librosa.stft(y)).mean(1)
    dominant = librosa.fft_frequencies(sr=sr)[spectrum[1:].argmax() + 1]  # skip DC

    # ponytail: pyin is slow, so pitch only looks at the first 30 s; use start= to look elsewhere
    f0, voiced, _ = librosa.pyin(y[: 30 * sr], fmin=50, fmax=2000, sr=sr)
    f0 = f0[voiced]

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    corr, key = max(
        (np.corrcoef(chroma, np.roll(prof, k))[0, 1], f"{NOTES[k]} {mode}")
        for mode, prof in (("major", MAJOR), ("minor", MINOR))
        for k in range(12)
    )
    return {
        "duration_s": r(len(y) / sr),
        "sample_rate": native_sr,
        "channels": channels,
        "peak_dbfs": db(peak),
        "rms_dbfs": db(rms),
        "clipped_pct": r(100 * np.mean(np.abs(raw) >= 0.999), 3),
        "silence_pct": r(100 * (1 - sum(e - s for s, e in sounding) / len(y)), 1),
        "sound_segments_s": [[r(start + s / sr, 1), r(start + e / sr, 1)] for s, e in sounding[:30]],
        "dominant_freq_hz": r(dominant, 1),
        "dominant_note": librosa.hz_to_note(dominant) if dominant > 0 else None,
        "spectral_centroid_hz": r(librosa.feature.spectral_centroid(y=y, sr=sr).mean(), 0),
        "pitch": {
            "median_hz": r(np.median(f0), 1),
            "median_note": librosa.hz_to_note(np.median(f0)),
            "range_notes": [librosa.hz_to_note(np.percentile(f0, 5)), librosa.hz_to_note(np.percentile(f0, 95))],
            "voiced_pct_first_30s": r(100 * voiced.mean(), 1),
        } if f0.size else None,
        "music": {
            "tempo_bpm": r(np.atleast_1d(tempo)[0], 1),
            "beats_detected": len(beats),
            "key": key,
            "key_confidence": r(corr),
            "note": "tempo/key are only meaningful if classify_sounds says this is music",
        },
    }


@mcp.tool(structured_output=False)
def spectrogram(path: str, start: float = 0.0, duration: float = 30.0) -> list:
    """Render the audio as an image you can look at: waveform (loudness over time) on top,
    mel spectrogram (frequency over time, brighter = louder) below."""
    sr = 22050
    y, _ = load(path, sr=sr, start=start, duration=duration)
    t = start + np.arange(len(y)) / sr
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)

    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 3])
    a.plot(t, y, lw=0.4)
    a.set(ylabel="amplitude", ylim=(-1, 1), title=f"{os.path.basename(path)}  {start:.1f}-{t[-1]:.1f} s")
    img = librosa.display.specshow(S, sr=sr, x_coords=start + librosa.times_like(S, sr=sr),
                                   x_axis="time", y_axis="mel", ax=b, cmap="magma")
    b.set_xlim(start, t[-1])  # waveform's autoscale margins would otherwise pad past the audio
    fig.colorbar(img, ax=[a, b], format="%+.0f dB")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return [Image(data=buf.getvalue(), format="png"),
            f"Window {start:.1f}-{t[-1]:.1f} s. Top: waveform. Bottom: mel spectrogram, dB relative to the loudest point."]


@cache
def _classifier():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    return AutoFeatureExtractor.from_pretrained(CLASSIFIER), AutoModelForAudioClassification.from_pretrained(CLASSIFIER).eval()


@cache
def _asr():
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=WHISPER)


if __name__ == "__main__":
    mcp.run()
