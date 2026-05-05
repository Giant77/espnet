"""
manifests_to_kaldi.py
...
UPDATED:
- Dry-run mode (--dry-run): no files written; prints summaries only
"""

import json
import os
import argparse
from collections import defaultdict


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
# Path Normalization
# ─────────────────────────────────────────────────────────────────────────────

def to_wsl_path(path: str) -> str:
    if not path:
        return path

    path = path.replace('\\', '/')

    if "dataset/processed/" in path:
        suffix = path.split("dataset/processed/", 1)[1]
        path = f"downloads/processed/{suffix}"

    if len(path) > 2 and path[1] == ':':
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize_records(records: list, name: str):
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
        print(f"  sample wav : {to_wsl_path(sample.get('wav_path',''))}")


# ─────────────────────────────────────────────────────────────────────────────
# Kaldi Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_kaldi_dir(records: list, output_dir: str, dry_run: bool = False) -> None:
    if dry_run:
        summarize_records(records, output_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    wav_scp = {}
    text_map = {}
    utt2spk = {}

    for rec in records:
        utt_id = rec['utt_id']
        wav_path = to_wsl_path(rec['wav_path'])
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

def split_records(records: list, train_ratio=0.8, dev_ratio=0.1, seed=42):
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

    manifest_dir = "downloads/processed/manifests/balanced"
    data_dir = "data"

    lang_groups = {
        "id": ["id_cv", "id_fleurs", 
            #    "id_librivox",
                 "id_titml", "id_indocsc", "id_sindodsc"],
        "ar": ["ar_cv", "ar_fleurs", "ar_clartts"],
        "en": ["en_librispeech", "en_fleurs", "en_cv_spon"],
        "cs": ["cs_escwa", "cs_hari", "cs_homostoria"],
    }

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

    # Write per-language
    for lang, splits in all_lang_splits.items():
        for split in ["train", "dev", "test"]:
            out_dir = os.path.join(data_dir, lang, split)
            write_kaldi_dir(splits[split], out_dir, dry_run=args.dry_run)

    # Trilingual
    tri = {"train": [], "dev": [], "test": []}
    for lang in ["id", "ar", "en"]:
        for split in ["train", "dev", "test"]:
            tri[split].extend(all_lang_splits[lang][split])

    for split in ["train", "dev", "test"]:
        out_dir = os.path.join(data_dir, "tri", split)
        write_kaldi_dir(tri[split], out_dir, dry_run=args.dry_run)
