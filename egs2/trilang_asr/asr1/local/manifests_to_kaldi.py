"""
manifests_to_kaldi.py
"""

import json
import os
import re
import csv
import string
import argparse
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def run_cmd(cmd):
    print("──"*25)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print()

# ─────────────────────────────────────────────────────────────────────────────
# Manifest Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_records(manifest: dict) -> dict:
    if isinstance(manifest, list):
        return {"all": manifest}

    splits = {}
    for split in ["train", "dev", "test"]:
        if split in manifest:
            splits[split] = manifest[split]

    if not splits:
        return {"all": manifest.get("records", [])}

    return splits

# ─────────────────────────────────────────────────────────────────────────────
# INF23_CS Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_inf23_cs_transcript(tsv_path: str) -> dict:
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

def parse_inf23_cs_wav(wav_path: str):
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

def load_inf23_cs(base_dir: str) -> list:
    """Loads INF23_CS as a flat list of records."""
    source_root = os.path.join(base_dir, "INF23_CS")
    tsv_path = os.path.join(source_root, "transcript.tsv")

    if not os.path.exists(tsv_path):
        print(f"WARN: Missing {tsv_path}")
        return []

    transcripts = load_inf23_cs_transcript(tsv_path)

    records = []
    seen_utt_ids = set()

    for root, _dirs, files in os.walk(source_root):
        for fn in sorted(files):
            if not fn.lower().endswith('.wav'):
                continue

            wav_path = os.path.join(root, fn)
            spk, audio_id = parse_inf23_cs_wav(wav_path)

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

            records.append({
                "utt_id": utt_id,
                "wav_path": wav_path.replace('\\', '/'),
                "text": text,
                "speaker": f"INF23_CS_{spk}",
            })

    return records

# ─────────────────────────────────────────────────────────────────────────────
# Path Normalization
# ─────────────────────────────────────────────────────────────────────────────
def to_wsl_path(path: str, base_dir: str = "downloads") -> str:
    if not path:
        return path

    path = path.replace('\\', '/')

    if "dataset/processed/" in path:
        suffix = path.split("dataset/processed/", 1)[1]
        path = f"{base_dir}/processed/{suffix}"

    if len(path) > 2 and path[1] == ':':
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"

    return path

# ─────────────────────────────────────────────────────────────────────────────
# Dry-run Summary
# ─────────────────────────────────────────────────────────────────────────────
def summarize_records(records: list, name: str, base_dir: str = "downloads"):
    n = len(records)
    speakers = set()
    total_dur = 0.0

    for r in records:
        speakers.add(r.get("speaker", "spk_unknown"))
        total_dur += float(r.get("duration", 0.0))

    print(f"[DRY-RUN] {name}")
    print(f"  utterances : {n}")
    print(f"  speakers   : {len(speakers)}")
    print(f"  hours      : {total_dur / 3600:.2f}")

    if n > 0:
        sample = records[0]
        print(f"  sample utt : {sample.get('utt_id')}")
        print(f"  sample wav : {to_wsl_path(sample.get('wav_path',''), base_dir=base_dir)}")

# ─────────────────────────────────────────────────────────────────────────────
# Kaldi Writer
# ─────────────────────────────────────────────────────────────────────────────
def write_kaldi_dir(records: list, output_dir: str, dry_run: bool = False, base_dir: str = "downloads") -> None:
    if dry_run:
        summarize_records(records, output_dir, base_dir=base_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    wav_scp = {}
    text_map = {}
    utt2spk = {}

    for rec in records:
        utt_id = rec['utt_id']
        wav_path = to_wsl_path(rec['wav_path'], base_dir=base_dir)
        text = rec['text']
        speaker = rec.get('speaker', f"spk_{utt_id[:8]}")

        wav_scp[utt_id] = wav_path
        text_map[utt_id] = text
        utt2spk[utt_id] = speaker

    sorted_utts = sorted(wav_scp.keys())

    with open(os.path.join(output_dir, 'wav.scp'), 'w', encoding='utf-8') as f:
        for u in sorted_utts:
            f.write(f"{u} {wav_scp[u]}\n")

    with open(os.path.join(output_dir, 'text'), 'w', encoding='utf-8') as f:
        for u in sorted_utts:
            f.write(f"{u} {text_map[u]}\n")

    with open(os.path.join(output_dir, 'utt2spk'), 'w', encoding='utf-8') as f:
        for u in sorted_utts:
            f.write(f"{u} {utt2spk[u]}\n")

    spk2utt = defaultdict(list)
    for u, s in utt2spk.items():
        spk2utt[s].append(u)

    with open(os.path.join(output_dir, 'spk2utt'), 'w', encoding='utf-8') as f:
        for s in sorted(spk2utt.keys()):
            f.write(f"{s} {' '.join(sorted(spk2utt[s]))}\n")

    print(f"Written {len(sorted_utts)} utterances → {output_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────
def split_records(records: list, train_ratio=0.8, dev_ratio=0.1, seed=777):
    import random
    random.seed(seed)

    spk_to_utts = defaultdict(list)
    for rec in records:
        spk_to_utts[rec.get('speaker', 'spk_unknown')].append(rec)

    speakers = sorted(spk_to_utts.keys())
    random.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_spks = set(speakers[:n_train])
    dev_spks = set(speakers[n_train:n_train + n_dev])
    test_spks = set(speakers[n_train + n_dev:])

    train = [r for r in records if r.get('speaker') in train_spks]
    dev = [r for r in records if r.get('speaker') in dev_spks]
    test = [r for r in records if r.get('speaker') in test_spks]

    return train, dev, test

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action='store_true')
    args = parser.parse_args()

    base_dir = "downloads"
    manifest_dir = os.path.join(base_dir, "processed", "manifests", "balanced")
    output_data_dir = "data"

    lang_groups = {
        "id": ["id_cv", "id_fleurs",
            #    "id_librivox",
                 "id_titml", "id_indocsc", "id_sindodsc"],
        "ar": ["ar_cv", "ar_fleurs", "ar_clartts"],
        "en": ["en_librispeech", "en_fleurs", "en_cv_spon"],
        "cs": ["cs_escwa", "cs_hari", "cs_homostoria"],
    }

    output_dirs = []
    all_lang_splits = {}

    for lang, keys in lang_groups.items():
        lang_splits = {"train": [], "dev": [], "test": [], "all": []}

        for key in keys:
            mf = os.path.join(manifest_dir, f"{key}_manifest.json")

            if not os.path.exists(mf):
                print(f"WARN: Missing {mf}")
                continue

            manifest = load_manifest(mf)
            splits = extract_records(manifest)

            if "train" in splits:
                for s in ["train", "dev", "test"]:
                    lang_splits[s].extend(splits.get(s, []))
            else:
                lang_splits["all"].extend(splits["all"])

        if lang_splits["all"] and not lang_splits["train"]:
            tr, dv, te = split_records(lang_splits["all"])
            lang_splits["train"] = tr
            lang_splits["dev"] = dv
            lang_splits["test"] = te

        all_lang_splits[lang] = lang_splits

    # ─── Extra test-only sources ───────────────────────────────────────────
    # These are appended directly to a split's "test" list and never pass
    # through split_records.
    extra_test_sources = {
        "cs": [load_inf23_cs],
    }

    for lang, loaders in extra_test_sources.items():
        if lang not in all_lang_splits:
            continue
        for loader in loaders:
            extra_records = loader(base_dir)
            print(f"{loader.__name__}: adding {len(extra_records)} utterances to {lang}/test")
            all_lang_splits[lang]["test"].extend(extra_records)

    # Write per-language
    for lang, splits in all_lang_splits.items():
        for split in ["train", "dev", "test"]:
            out_dir = os.path.join(output_data_dir, lang, split)
            write_kaldi_dir(splits[split], out_dir, dry_run=args.dry_run, base_dir=base_dir)
            output_dirs.append(out_dir)

    # Trilingual
    tri = {"train": [], "dev": [], "test": []}
    for lang in ["id", "ar", "en"]:
        for split in ["train", "dev", "test"]:
            tri[split].extend(all_lang_splits[lang][split])

    for split in ["train", "dev", "test"]:
        out_dir = os.path.join(output_data_dir, "tri", split)
        write_kaldi_dir(tri[split], out_dir, dry_run=args.dry_run, base_dir=base_dir)
        output_dirs.append(out_dir)

# fix + validate
if not args.dry_run:
    for d in output_dirs:
        run_cmd(["./utils/fix_data_dir.sh", d])
        run_cmd(["./utils/validate_data_dir.sh", d, "--no-feats"])
