"""
manifests_to_kaldi.py
Converts JSON manifests (from Windows preprocessing) to Kaldi-style files:
  wav.scp, text, utt2spk, spk2utt
NEW: Using absolute WSL paths; supports multi-dataset merge per language
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_manifest(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_kaldi_dir(records: list, output_dir: str,
                     remap_audio_root: str = None) -> None:
    """
    Write Kaldi data directory files.
    remap_audio_root: if set, replaces Windows-style paths with WSL path
    """
    os.makedirs(output_dir, exist_ok=True)

    wav_scp   = {}
    text_map  = {}
    utt2spk   = {}

    for rec in records:
        utt_id   = rec['utt_id']
        wav_path = rec['wav_path']
        text     = rec['text']
        speaker  = rec.get('speaker', f"spk_{utt_id[:8]}")

        if remap_audio_root:
            # Convert Windows path to WSL path
            wav_path = wav_path.replace('\\', '/')
            drive = wav_path[:2]
            if ':' in drive:
                drive_letter = drive[0].lower()
                wav_path = f"/mnt/{drive_letter}" + wav_path[2:]

        wav_scp[utt_id]  = wav_path
        text_map[utt_id] = text
        utt2spk[utt_id]  = speaker

    # Sort all by utterance ID (Kaldi requirement)
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

    # Generate spk2utt from utt2spk
    spk2utt = defaultdict(list)
    for u, s in utt2spk.items():
        spk2utt[s].append(u)

    with open(os.path.join(output_dir, 'spk2utt'), 'w', encoding='utf-8') as f:
        for s in sorted(spk2utt.keys()):
            f.write(f"{s} {' '.join(sorted(spk2utt[s]))}\n")

    print(f"Written {len(sorted_utts)} utterances to {output_dir}")


def split_records(records: list,
                   train_ratio: float = 0.7,
                   dev_ratio: float   = 0.2,
                   seed: int          = 42) -> tuple:
    """
    Speaker-independent 7:2:1 split.
    NEW: Fixed seed for reproducibility — mandatory for TA submission
    """
    import random
    random.seed(seed)

    # Group by speaker
    spk_to_utts = defaultdict(list)
    for rec in records:
        spk_to_utts[rec.get('speaker', 'spk_unknown')].append(rec)

    speakers = sorted(spk_to_utts.keys())
    random.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)

    train_spks = set(speakers[:n_train])
    dev_spks   = set(speakers[n_train:n_train + n_dev])
    test_spks  = set(speakers[n_train + n_dev:])

    train = [r for r in records if r.get('speaker') in train_spks]
    dev   = [r for r in records if r.get('speaker') in dev_spks]
    test  = [r for r in records if r.get('speaker') in test_spks]

    return train, dev, test


if __name__ == '__main__':
    manifest_dir = "downloads/processed/manifests"
    data_dir     = "data"
    wsl_proc_root = "downloads/processed"

    # ─── Language groups ─────────────────────────────────────────────────────
    lang_groups = {
        "id":  ["id_cv", "id_fleurs", "id_librivox", "id_titml",
                 "id_indocsc", "id_sindodsc"],
        "ar":  ["ar_cv", "ar_fleurs", "ar_clartts"],
        "en":  ["en_librispeech", "en_fleurs", "en_cv_spon"],
        "cs":  ["cs_escwa", "cs_hari", "cs_homostoria"],
    }

    all_lang_records = {}

    for lang, keys in lang_groups.items():
        combined = []
        for key in keys:
            mf = os.path.join(manifest_dir, f"{key}_manifest.json")
            if os.path.exists(mf):
                recs = load_manifest(mf)
                combined.extend(recs)
                print(f"  Loaded {key}: {len(recs)} records")
            else:
                print(f"  WARN: Missing manifest {mf}")
        all_lang_records[lang] = combined
        print(f"  {lang.upper()} total: {len(combined)} records")

    # ─── Per-language splits and Kaldi dirs ──────────────────────────────────
    for lang, records in all_lang_records.items():
        if lang == "cs":
            # CS: split independently with same 7:2:1 ratio
            train, dev, test = split_records(records)
        else:
            train, dev, test = split_records(records)

        for split, recs in [('train', train), ('dev', dev), ('test', test)]:
            out_dir = os.path.join(data_dir, lang, split)
            write_kaldi_dir(recs, out_dir)

    # ─── Trilingual combined (tri = id + ar + en, no cs) ─────────────────────
    tri_records = (all_lang_records['id'] +
                   all_lang_records['ar'] +
                   all_lang_records['en'])
    print(f"\nTrilingual total: {len(tri_records)} records")
    train_tri, dev_tri, test_tri = split_records(tri_records)
    for split, recs in [('train', train_tri), ('dev', dev_tri), ('test', test_tri)]:
        out_dir = os.path.join(data_dir, "tri", split)
        write_kaldi_dir(recs, out_dir)

    print("\nKaldi directories written. Run utils/fix_data_dir.sh next.")