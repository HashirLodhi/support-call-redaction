"""Synthesise a stereo mock support call from data/script.csv.
Agent -> left channel, Customer -> right channel (like real dual-channel call recordings).
Skip this script if you record a real call: save it as audio/call.wav
(stereo, one speaker per channel) and run pipeline.py."""
import csv, json, os, subprocess, tempfile, pathlib
from pydub import AudioSegment

ROOT = pathlib.Path(__file__).parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
rows = list(csv.DictReader(open(ROOT / "data/script.csv", encoding="utf-8")))
SR = 16000


def tts(text, voice, out):
    ps = ("Add-Type -AssemblyName System.Speech;"
          "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          f"$s.SelectVoice('{voice}');$s.Rate=0;"
          f"$s.SetOutputToWaveFile('{out}');$s.Speak($env:LINE);$s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-c", ps], check=True,
                   env={**os.environ, "LINE": text})


def silence(ms):
    return AudioSegment.silent(ms, frame_rate=SR)


gap = int(cfg["gap_seconds"] * 1000)
left, right = silence(0), silence(0)
with tempfile.TemporaryDirectory() as tmp:
    for i, r in enumerate(rows):
        wav = f"{tmp}/{i}.wav"
        tts(r["text"], cfg["voices"][r["speaker"]], wav)
        seg = AudioSegment.from_wav(wav).set_frame_rate(SR).set_channels(1)
        quiet = silence(len(seg) + gap)
        talk = seg + silence(gap)
        if r["speaker"] == "Agent":
            left, right = left + talk, right + quiet
        else:
            left, right = left + quiet, right + talk

n = max(len(left.get_array_of_samples()), len(right.get_array_of_samples()))
pad = lambda a: a + AudioSegment.silent(int((n - len(a.get_array_of_samples())) / SR * 1000) + 1, SR)
left, right = pad(left), pad(right)
m = min(len(left.get_array_of_samples()), len(right.get_array_of_samples()))
left, right = left._spawn(left.raw_data[: m * 2]), right._spawn(right.raw_data[: m * 2])
call = AudioSegment.from_mono_audiosegments(left, right)
(ROOT / "audio").mkdir(exist_ok=True)
call.export(ROOT / "audio/call.wav", format="wav")
print(f"audio/call.wav  {len(call)/1000:.1f}s  ({len(rows)} lines)")
