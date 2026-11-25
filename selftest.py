"""End-to-end check: spawns server.py over stdio MCP and calls every tool on generated audio.
First run downloads the models (~500 MB). Speech fixture uses Windows' built-in TTS."""

import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from mcp import Client
from mcp.client.stdio import StdioServerParameters

SPEECH = (
    "The quick brown fox jumps over the lazy dog. "
    "Sound understanding lets a language model know what is happening in a recording. "
    "It can tell speech from music, find the words that were spoken, and measure how loud things are. "
    "Whisper handles audio longer than thirty seconds by working through it piece by piece. "
    "We keep talking for a while so the recording runs past that limit. "
    "Rain on a window, a dog barking, and a car horn are all sounds it should recognise. "
    "This is the final sentence of the test."
)


def fixtures(d):
    sr = 22050
    t = np.arange(3 * sr) / sr
    tone = os.path.join(d, "tone.wav")  # 3 s of A4 then 1 s of silence
    sf.write(tone, np.concatenate([0.5 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)]), sr)

    music = os.path.join(d, "music.wav")  # C-E-G-E arpeggio, one note per beat at 120 BPM, 12 s
    beat = np.arange(sr // 2) / sr
    notes = [261.63, 329.63, 392.0, 329.63] * 6
    sig = np.concatenate([np.sin(2 * np.pi * f * beat) * np.exp(-6 * beat) for f in notes])
    sf.write(music, 0.6 * sig, sr)

    speech = os.path.join(d, "speech.wav")
    ps = (
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SelectVoiceByHints('NotSet', 'NotSet', 0, [Globalization.CultureInfo]'en-US'); "  # default may be non-English
        f"$s.Rate = -1; $s.SetOutputToWaveFile('{speech}'); $s.Speak('{SPEECH}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return tone, music, speech


async def main():
    with tempfile.TemporaryDirectory() as d:
        tone, music, speech = fixtures(d)
        print("speech fixture:", round(sf.info(speech).duration, 1), "s")
        here = os.path.dirname(os.path.abspath(__file__))
        params = StdioServerParameters(command=sys.executable, args=[os.path.join(here, "server.py")])
        async with Client(params, read_timeout_seconds=1800) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"classify_sounds", "transcribe", "analyze_audio", "spectrogram"}, names

            async def call(name, **args):
                res = await c.call_tool(name, args)
                assert not res.is_error, res.content
                return res

            def data(res):
                return json.loads(res.content[0].text)

            a = data(await call("analyze_audio", path=tone))
            print("tone:", json.dumps(a))
            assert abs(a["dominant_freq_hz"] - 440) < 15 and a["dominant_note"] == "A4", a
            assert 20 < a["silence_pct"] < 30 and a["pitch"]["median_note"] == "A4", a

            m = data(await call("analyze_audio", path=music))
            print("music:", json.dumps(m["music"]))
            assert 110 < m["music"]["tempo_bpm"] < 130, m["music"]

            cl = data(await call("classify_sounds", path=speech))
            print("classify:", json.dumps(cl["top_sounds"][:3]))
            assert cl["top_sounds"][0]["label"] == "Speech", cl
            assert len(cl["timeline"]) >= 4, cl["timeline"]

            tr = data(await call("transcribe", path=speech))
            print("transcript:", tr["text"])
            low = tr["text"].lower()
            assert "quick brown fox" in low and "final sentence" in low, low  # last line is past 30 s
            assert tr["segments"] and tr["segments"][-1]["start_s"] > 30, tr["segments"][-1]

            sp = await call("spectrogram", path=speech, start=5, duration=10)
            img = sp.content[0]
            assert img.type == "image" and base64.b64decode(img.data)[:4] == b"\x89PNG", img.type
            with open(os.path.join(here, "selftest_spectrogram.png"), "wb") as f:
                f.write(base64.b64decode(img.data))
    print("OK: all 4 tools pass")


if __name__ == "__main__":
    asyncio.run(main())
