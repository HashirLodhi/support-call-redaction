"""Transcribe -> label speakers -> timestamp -> redact PII -> tag purpose -> evaluate.

Usage: python pipeline.py [path/to/stereo_call.wav]
Channel 0 = Agent, channel 1 = Customer.
"""
import csv, json, re, sys, pathlib
from faster_whisper import WhisperModel
from pydub import AudioSegment

ROOT = pathlib.Path(__file__).parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
AUDIO = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audio/call.wav"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


# ---------- 1. transcription (one pass per channel => reliable Agent/Customer labels) ----------
def transcribe():
    stereo = AudioSegment.from_file(AUDIO)
    model = WhisperModel(cfg["whisper_model"], compute_type="int8")
    lines = []
    for label, ch in zip(["Agent", "Customer"], stereo.split_to_mono()):
        tmp = OUT / f"_{label}.wav"
        ch.set_frame_rate(16000).export(tmp, format="wav")
        segs, _ = model.transcribe(str(tmp), language="en", vad_filter=True, beam_size=5,
                                   word_timestamps=True)
        for s in segs:
            if s.words:  # word timings are tighter than segment timings (no leading silence)
                lines.append({"speaker": label, "start": round(s.words[0].start, 2),
                              "end": round(s.words[-1].end, 2), "raw": s.text.strip()})
        tmp.unlink()
    lines.sort(key=lambda l: l["start"])
    # merge consecutive segments from the same speaker into one turn
    turns = []
    for l in lines:
        if turns and turns[-1]["speaker"] == l["speaker"]:
            turns[-1]["end"] = l["end"]
            turns[-1]["raw"] += " " + l["raw"]
        else:
            turns.append(dict(l))
    return turns


# ---------- 2. PII redaction ----------
NAME_CTX = re.compile(
    r"\b(?:my name is|this is|i'm|i am|speaking with|call me|thanks?,?|thank you,?|"
    r"sorry to hear that,?|welcome,?|hi,?|hello,?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
ORDER = re.compile(r"(?i)\b(order(?:\s+(?:number|id|no\.?|#))?\s*(?:is|:|#)?\s*)([A-Z]{0,4}-?\d[\d,\s-]{3,}\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def redact(text):
    found = []

    def sub(pattern, tag, s, keep_prefix=False):
        def f(m):
            found.append(tag)
            return (m.group(1) if keep_prefix else "") + f"[{tag}]"
        return pattern.sub(f, s)

    t = sub(EMAIL, "EMAIL", text)
    t = sub(CARD, "CARD", t)
    t = sub(ORDER, "ORDER_ID", t, keep_prefix=True)   # before phone: order ids are shorter
    t = sub(PHONE, "PHONE", t)

    def name_ctx(m):
        found.append("NAME")
        return m.group(0).replace(m.group(1), "[NAME]")

    t = NAME_CTX.sub(name_ctx, t)
    for n in cfg["known_names"]:                      # known-name list catches the rest
        t, k = re.subn(rf"\b{re.escape(n)}\b", "[NAME]", t, flags=re.I)
        found += ["NAME"] * k
    t = re.sub(r"(\[NAME\])(?:\s+\[NAME\])+", r"\1", t)
    return t, sorted(set(found))


# ---------- 3. purpose tagging (rule-based, first match wins) ----------
RULES = [
    ("refund_request", r"refund.*(like|want|need|please)|(like|want|need).*refund|money back"),
    ("resolution", r"issued|processed|has been (refunded|resolved)|reach your card"),
    ("complaint", r"cracked|broken|damaged|defective|not working|arrived with|never arrived|problem"),
    ("identity_verification", r"order number|phone number|confirm it'?s you|verify|date of birth"),
    ("provide_info", r"\[order_id\]|\[phone\]"),
    ("closing", r"goodbye|good day|anything else|that'?s everything|you'?re welcome|bye"),
    ("greeting", r"thank you for calling|how can i help|^hi\b|^hello\b|my name is"),
    ("confirmation", r"got it|i can see|thank you|great|thanks"),
]


def tag(text):
    low = text.lower()
    for name, pat in RULES:
        if re.search(pat, low):
            return name
    return "other"


# ---------- 4. evaluation against the script ----------
def norm(s):
    return re.findall(r"[a-z0-9\[\]_']+", s.lower())


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def ts(x):
    return f"{int(x // 60):02d}:{x % 60:05.2f}"


def main():
    lines = transcribe()
    for l in lines:
        l["text"], l["pii_found"] = redact(l["raw"])
        l["purpose"] = tag(l["text"])

    # raw (unredacted) text is deliberately NOT written to the published outputs
    public = [{k: v for k, v in l.items() if k != "raw"} for l in lines]
    (OUT / "transcript_redacted.json").write_text(json.dumps(public, indent=1))
    with open(OUT / "transcript_redacted.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker", "purpose", "text"])
        for l in public:
            w.writerow([ts(l["start"]), ts(l["end"]), l["speaker"], l["purpose"], l["text"]])
    md = ["# Redacted support call transcript\n", "| Time | Speaker | Purpose | Text |", "|---|---|---|---|"]
    md += [f"| {ts(l['start'])} | {l['speaker']} | `{l['purpose']}` | {l['text']} |" for l in public]
    (OUT / "transcript_redacted.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "_raw_transcript_PRIVATE.json").write_text(json.dumps(lines, indent=1))  # gitignored

    gold_csv = ROOT / "data/script.csv"
    if gold_csv.exists():
        script = list(csv.DictReader(open(gold_csv, encoding="utf-8")))
        gold = []  # merge consecutive same-speaker script lines into turns, like the pipeline does
        for g in script:
            if gold and gold[-1]["speaker"] == g["speaker"]:
                gold[-1]["text"] += " " + g["text"]
                gold[-1]["purposes"].add(g["purpose"])
            else:
                gold.append({"speaker": g["speaker"], "text": g["text"], "purposes": {g["purpose"]}})
        raw_all = " ".join(l["raw"] for l in lines)
        ref_all = " ".join(g["text"] for g in gold)
        red_all = " ".join(l["text"] for l in lines)
        pii = ["Jordan", "Mercer", "Priya", "482913", "482,913", "555", "014", "2278"]
        leaks = [p for p in pii if p.lower() in red_all.lower()]
        pur_ok = sum(1 for l, g in zip(lines, gold) if l["purpose"] in g["purposes"])
        spk_ok = sum(1 for l, g in zip(lines, gold) if l["speaker"] == g["speaker"])
        rep = [
            "# Evaluation", "",
            f"- Lines transcribed: {len(lines)} (script has {len(gold)})",
            f"- Transcription WER vs script: {wer(ref_all, raw_all):.1%}",
            f"- Speaker label accuracy: {spk_ok}/{len(gold)}",
            f"- Purpose tag accuracy: {pur_ok}/{len(gold)}",
            f"- PII strings still present after redaction: {leaks or 'none'}",
            f"- PII tags used: {sorted({t for l in lines for t in l['pii_found']})}",
        ]
        (OUT / "evaluation.md").write_text("\n".join(rep) + "\n")
        print("\n".join(rep) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
