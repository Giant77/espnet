import os
import re
import csv
import json
import string
import unicodedata
import subprocess
from pyarabic import araby
from tqdm import tqdm

CONFIG = {
    "sample_rate": 16000,
    "min_duration_sec": 1.0,
    "max_duration_sec": 30.0,
    "target_duration_sec": 10.0, # target for segmentation
    "silence_threshold_db": -50,
    "silence_min_len_ms": 500,
}

# ─────────────────────────────────────────────────────────────────────────────
# generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Convert any audio to 16kHz mono WAV using ffmpeg.

    Returns:
        bool: True if conversion succeeds, False otherwise.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(CONFIG["sample_rate"]),
        "-ac", "1",  # mono
        "-acodec", "pcm_s16le",
        output_path,
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        stderr_msg = result.stderr.decode().strip() if result.stderr else "No stderr output"
        print(f"Error: ffmpeg failed to convert {input_path} to {output_path}. stderr: {stderr_msg}")
        return False
    return True

def save_manifest(records, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def load_manifest_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# text normalization (stage 2)
# ─────────────────────────────────────────────────────────────────────────────
def remove_invisible_chars(text: str) -> str:
    return ''.join(
        c for c in text
        if unicodedata.category(c) != 'Cf'
    )

def remove_transcript_tags(text: str) -> str:
    return re.sub(r'\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>', ' ', text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize_arabic(text: str, araby=araby) -> str:
    if araby:
        # Remove harakat/tashkeel
        text = araby.strip_diacritics(text)

        # Normalize lam-alef ligatures
        text = araby.normalize_ligature(text)

    # Remove tatweel
    text = text.replace('ـ', '')
    text = normalize_whitespace(text)

    return text

def contains_arabic(text: str) -> bool:
    return any(
        '\u0600' <= c <= '\u06FF'
        or '\u0750' <= c <= '\u077F'
        or '\u08A0' <= c <= '\u08FF'
        or '\uFB50' <= c <= '\uFDFF'
        or '\uFE70' <= c <= '\uFEFF'
        for c in text
    )

def normalize_common(text: str) -> str:
    text = remove_transcript_tags(text)
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    text = remove_invisible_chars(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text

def normalize_transcript(text: str) -> str:
    """Full pipeline: common normalization, then Arabic-specific if detected."""
    text = normalize_common(text)
    if contains_arabic(text):
        text = normalize_arabic(text)
    return text

# ─────────────────────────────────────────────────────────────────────────────
# INF23_CS helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_inf23_cs_transcript(tsv_path: str) -> dict:
    """Headerless TSV: <id>\\t<transcript>. Returns {str(int(id)): text}."""
    transcripts = {}
    with open(tsv_path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row or len(row) < 2:
                continue

            utt_key = row[0].strip()
            text = row[1].strip()

            if not utt_key:
                continue

            try:
                transcripts[str(int(utt_key))] = text
            except ValueError:
                print(f"WARN: non-numeric id in {tsv_path}: {row[0]!r}")

    return transcripts

def _parse_inf23_cs_wav(wav_path: str):
    """
    Returns (speaker_id, audio_id) for a wav file under INF23_CS.
    Filename (basename, regardless of nesting depth) is always
    "[spkid]_audio[X].wav".
    """
    wav_name_re = re.compile(r'^(?P<spk>\d+)_audio(?P<audio_id>\d+)', re.IGNORECASE)
    basename = os.path.splitext(os.path.basename(wav_path))[0]

    m = wav_name_re.match(basename)
    if not m:
        return None, None

    return m.group('spk'), str(int(m.group('audio_id')))

# ─────────────────────────────────────────────────────────────────────────────
# INF23_CS — stage 1 (audio)
# ─────────────────────────────────────────────────────────────────────────────
def stage1_inf23_cs(base_dir: str) -> list:
    """
    Converts INF23_CS audio to 16kHz mono wav, flat (no lang_codes/pair
    subdirs). Writes wavs to downloads/INF23_CS/processed/wavs/ and a
    stage1 manifest (utt_id, wav_path, speaker, raw_text) to
    downloads/INF23_CS/processed/manifests/stage1.json.
    """
    source_root = os.path.join(base_dir, "INF23_CS")
    tsv_path = os.path.join(source_root, "transcript.tsv")

    out_root = os.path.join(source_root, "processed")
    out_wav_dir = os.path.join(out_root, "wavs")
    manifest_path = os.path.join(out_root, "manifests", "stage1.json")
    os.makedirs(out_wav_dir, exist_ok=True)

    if not os.path.exists(tsv_path):
        print(f"WARN: Missing {tsv_path}")
        save_manifest([], manifest_path)
        return []

    transcripts = _load_inf23_cs_transcript(tsv_path)

    wav_files = []
    out_root_abs = os.path.abspath(out_root)
    for root, dirs, files in os.walk(source_root):
        # Don't descend into our own output dir on re-runs
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != out_root_abs]
        for fn in sorted(files):
            if fn.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, fn))

    records = []
    seen_utt_ids = set()
    skipped = 0

    for wav_path in tqdm(wav_files, desc="Stage 1: INF23_CS audio"):
        spk, audio_id = _parse_inf23_cs_wav(wav_path)

        if spk is None or audio_id is None:
            print(f"WARN: Could not parse speaker/audio id from {wav_path}")
            continue

        text = transcripts.get(audio_id)
        if text is None:
            print(f"WARN: No transcript for id={audio_id} ({wav_path})")
            continue

        utt_id = f"INF23_CS_{spk}_audio{audio_id}"
        if utt_id in seen_utt_ids:
            utt_id = f"{utt_id}_{len(seen_utt_ids)}"
        seen_utt_ids.add(utt_id)

        out_wav = os.path.join(out_wav_dir, f"{utt_id}.wav")
        if not convert_to_wav(wav_path, out_wav):
            skipped += 1
            continue

        records.append({
            "utt_id": utt_id,
            "wav_path": out_wav.replace('\\', '/'),
            "raw_text": text,
            "speaker": f"INF23_CS_{spk}",
        })

    if skipped:
        print(f"stage1_inf23_cs: {skipped} files skipped (conversion failure)")

    save_manifest(records, manifest_path)
    print(f"stage1_inf23_cs: wrote {len(records)} records -> {manifest_path}")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# CS_MMS — stage 1 (audio)
# ─────────────────────────────────────────────────────────────────────────────
_MMS_SPLIT_FILES = {
    "train": "train.csv",
    "dev": "val.csv",
    "test": "test.csv",
}

def _stage1_cs_mms_split(pair_dir: str, pair: str, split: str, csv_name: str,
                          out_wav_dir: str) -> list:
    """
    Converts audio for one split CSV (train.csv / val.csv / test.csv) of a
    single lang-pair dir under cs_mms/<pair>/.
    CSV columns: sentence_id,audio_path,sentence,duration_s,pair,n_segments
    """
    csv_path = os.path.join(pair_dir, csv_name)
    if not os.path.exists(csv_path):
        print(f"WARN: Missing {csv_path}")
        return []

    with open(csv_path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in tqdm(rows, desc=f"Stage 1: CS_MMS {pair}/{csv_name}"):
        sentence_id = (row.get('sentence_id') or '').strip()
        audio_rel_path = (row.get('audio_path') or '').strip()
        text = (row.get('sentence') or '').strip()

        if not sentence_id or not audio_rel_path or not text:
            continue

        src_wav = os.path.join(pair_dir, audio_rel_path)
        if not os.path.exists(src_wav):
            print(f"WARN: Missing audio {src_wav}")
            continue

        utt_id = f"mms_{sentence_id}"
        out_wav = os.path.join(out_wav_dir, f"{utt_id}.wav")

        if not convert_to_wav(src_wav, out_wav):
            continue

        records.append({
            "utt_id": utt_id,
            "wav_path": out_wav.replace('\\', '/'),
            "raw_text": text,
            "speaker": sentence_id,
            "pair": pair,
            "split": split,
        })

    return records

def stage1_cs_mms(base_dir: str) -> list:
    """
    Converts CS_MMS audio (cs_mms/<pair>/{train,val,test}.csv) to
    16kHz mono wav. Writes wavs to
    downloads/cs_mms/processed/<pair>/wavs/ and a single combined
    stage1 manifest (all pairs/splits) to
    downloads/cs_mms/processed/manifests/stage1.json.
    metadata.csv is ignored.
    """
    source_root = os.path.join(base_dir, "cs_mms")

    out_root = os.path.join(base_dir, "cs_mms", "processed")
    manifest_path = os.path.join(out_root, "manifests", "stage1.json")

    if not os.path.isdir(source_root):
        print(f"WARN: Missing {source_root}")
        save_manifest([], manifest_path)
        return []

    records = []
    for pair in sorted(os.listdir(source_root)):
        pair_dir = os.path.join(source_root, pair)
        if not os.path.isdir(pair_dir):
            continue

        out_wav_dir = os.path.join(out_root, pair, "wavs")
        os.makedirs(out_wav_dir, exist_ok=True)

        for split, csv_name in _MMS_SPLIT_FILES.items():
            recs = _stage1_cs_mms_split(pair_dir, pair, split, csv_name, out_wav_dir)
            print(f"stage1_cs_mms: {pair}/{csv_name}: {len(recs)} utterances")
            records.extend(recs)

    save_manifest(records, manifest_path)
    print(f"stage1_cs_mms: wrote {len(records)} records -> {manifest_path}")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# stage 2 (transcript normalization) — dataset-agnostic
# ─────────────────────────────────────────────────────────────────────────────
def stage2_normalize_dataset(stage1_manifest_path: str, out_manifest_path: str,
                              desc: str = "Stage 2: normalizing transcripts") -> list:
    """
    Reads a stage1 manifest (utt_id, wav_path, raw_text, speaker, ...),
    applies normalize_transcript() to raw_text, and writes the final
    per-record manifest (utt_id, wav_path, text, speaker, ...) used by
    stage 3 / write_kaldi_dir.
    """
    if not os.path.exists(stage1_manifest_path):
        print(f"WARN: Missing stage1 manifest {stage1_manifest_path}")
        save_manifest([], out_manifest_path)
        return []

    stage1_records = load_manifest_json(stage1_manifest_path)

    records = []
    dropped = 0
    for rec in tqdm(stage1_records, desc=desc):
        raw_text = rec.get("raw_text", "")
        text = normalize_transcript(raw_text)

        if not text:
            dropped += 1
            continue

        out_rec = dict(rec)
        out_rec.pop("raw_text", None)
        out_rec["text"] = text
        records.append(out_rec)

    if dropped:
        print(f"stage2_normalize_dataset: {dropped} records dropped (empty after normalization)")

    save_manifest(records, out_manifest_path)
    print(f"stage2_normalize_dataset: wrote {len(records)} records -> {out_manifest_path}")
    return records

def stage2_inf23_cs(base_dir: str) -> list:
    out_root = os.path.join(base_dir, "INF23_CS", "processed", "manifests")
    return stage2_normalize_dataset(
        stage1_manifest_path=os.path.join(out_root, "stage1.json"),
        out_manifest_path=os.path.join(out_root, "records.json"),
        desc="Stage 2: INF23_CS transcripts",
    )

def stage2_cs_mms(base_dir: str) -> list:
    out_root = os.path.join(base_dir, "cs_mms", "processed", "manifests")
    return stage2_normalize_dataset(
        stage1_manifest_path=os.path.join(out_root, "stage1.json"),
        out_manifest_path=os.path.join(out_root, "records.json"),
        desc="Stage 2: CS_MMS transcripts",
    )

# ─────────────────────────────────────────────────────────────────────────────
# stage 2 output loaders — used by stage 3 / manifests_to_kaldi.py
# ─────────────────────────────────────────────────────────────────────────────
def load_inf23_cs_records(base_dir: str) -> list:
    """Loads stage2 output for INF23_CS as a flat list of records."""
    path = os.path.join(base_dir, "INF23_CS", "processed", "manifests", "records.json")
    if not os.path.exists(path):
        print(f"WARN: Missing {path} (run stage 1+2 first)")
        return []
    return load_manifest_json(path)

def load_cs_mms_records(base_dir: str) -> dict:
    """
    Loads stage2 output for CS_MMS, regrouped into
    {"train": [...], "dev": [...], "test": [...]} using each record's
    "split" field (preserves the dataset's own pre-existing split).
    """
    path = os.path.join(base_dir, "cs_mms", "processed", "manifests", "records.json")
    splits = {"train": [], "dev": [], "test": []}
    if not os.path.exists(path):
        print(f"WARN: Missing {path} (run stage 1+2 first)")
        return splits

    for rec in load_manifest_json(path):
        split = rec.get("split", "train")
        splits.setdefault(split, []).append(rec)
    return splits
