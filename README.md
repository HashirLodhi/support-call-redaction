# Support Call Transcription + PII Redaction

Call-center AI demo: transcribe a customer support call, label speakers, timestamp each turn,
redact personal data (`[NAME]`, `[PHONE]`, `[ORDER_ID]`), tag each turn's purpose, and measure accuracy.

## Pipeline
1. **Audio** – stereo WAV, Agent on the left channel, Customer on the right (standard dual-channel call recording).
2. **Transcribe** – `faster-whisper` per channel with word timestamps, so speaker labels come from the channel (no diarization guesswork). Consecutive segments from one speaker are merged into turns.
3. **Redact** – regex for phone / order number / email / card, context patterns ("my name is X"), plus a known-names list in `config.json`.
4. **Tag purpose** – rule-based: greeting, complaint, identity_verification, provide_info, refund_request, resolution, confirmation, closing.
5. **Evaluate** – WER vs the script, speaker accuracy, purpose accuracy, and a leak check for the fake PII strings.

## Run
```
pip install -r requirements.txt
python make_mock_audio.py     # synthetic call from data/script.csv (Windows SAPI voices)
python pipeline.py            # or: python pipeline.py path/to/your_call.wav
```
To use a real recording with a friend (with consent), script it with fake details, record one speaker per channel,
save as `audio/call.wav`, and update `known_names` in `config.json`.

## Outputs (`output/`)
`transcript_redacted.md|csv|json`, `evaluation.md`. The unredacted `_raw_transcript_PRIVATE.json` is gitignored – never publish it.

## Result on the mock call
WER 2.6%, speaker labels 13/13, purpose tags 12/13, 0 PII strings leaked.

## Limits
Redaction is pattern-based; unusual names or spoken-out digits ("five five five...") can slip through, so a production system
would add an NER model and human review. Purpose tagging is keyword rules, replaceable with an LLM classifier.
