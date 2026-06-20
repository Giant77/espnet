"""
manifests_to_kaldi.py

Staged pipeline:
  Stage 1 — preprocess INF23_CS and cs_mms audio (convert/resample to wav)
  Stage 2 — preprocess INF23_CS and cs_mms transcripts (normalize text)
  Stage 3 — convert manifests (existing lang_groups + stage2 CS outputs) to
            Kaldi data dirs

Each stage can be run independently and resumed via --stage / --stop_stage.
Stage 1 and 2 persist their output as JSON manifests on disk, so re-running
stage 2 (for example) does not require re-running stage 1.
"""

import json
import os
import argparse
import subprocess
from collections import defaultdict
from preprocess_cs import (
    stage1_inf23_cs,
    stage1_cs_mms,
    stage2_inf23_cs,
    stage2_cs_mms,
    load_inf23_cs_records,
    load_cs_mms_records,
)

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
# Stage 1 — audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(base_dir: str) -> None:
    print("=" * 70)
    print("STAGE 1: preprocessing INF23_CS and cs_mms audio")
    print("=" * 70)
    stage1_inf23_cs(base_dir)
    stage1_cs_mms(base_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — transcript preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def run_stage2(base_dir: str) -> None:
    print("=" * 70)
    print("STAGE 2: preprocessing INF23_CS and cs_mms transcripts")
    print("=" * 70)
    stage2_inf23_cs(base_dir)
    stage2_cs_mms(base_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — manifests to kaldi
# ─────────────────────────────────────────────────────────────────────────────
def run_stage3(base_dir: str, manifest_dir: str, output_data_dir: str,
               cs_mms_data_dir: str, use_mms: bool, dry_run: bool) -> list:
    print("=" * 70)
    print("STAGE 3: converting manifests to Kaldi data dirs")
    print("=" * 70)

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
    # through split_records. Reads stage2 output (records.json); run
    # stage 1+2 first.
    extra_test_sources = {
        "cs": [load_inf23_cs_records],
    }

    for lang, loaders in extra_test_sources.items():
        if lang not in all_lang_splits:
            continue
        for loader in loaders:
            extra_records = loader(base_dir)
            print(f"{loader.__name__}: adding {len(extra_records)} utterances to {lang}/test")
            all_lang_splits[lang]["test"].extend(extra_records)

    # ─── cs_mms: always written standalone; optionally folded into data/cs ──
    mms_splits = load_cs_mms_records(base_dir)
    # for split in ["train", "dev", "test"]:
    #     out_dir = os.path.join(cs_mms_data_dir, split)
    #     write_kaldi_dir(mms_splits[split], out_dir, dry_run=dry_run, base_dir=base_dir)
    #     output_dirs.append(out_dir)

    # if use_mms:
        # mms_splits[split].extend(all_lang_splits["cs"][split])
        # for split in ["train", "dev", "test"]:
        #     print(f"load_cs_mms_records: folding {len(mms_splits[split])} utterances into cs/{split}")
        #     all_lang_splits["cs"][split].extend(mms_splits[split])
    if use_mms:
        for split in ["train", "dev", "test"]:
            print(f"run_stage3: folding {len(all_lang_splits['cs'][split])} primary cs utterances into cs_mms/{split}")
    
            out_dir = os.path.join(cs_mms_data_dir, split)
            mms_splits[split].extend(all_lang_splits["cs"][split])

            write_kaldi_dir(mms_splits[split], out_dir, dry_run=dry_run, base_dir=base_dir)
            output_dirs.append(out_dir)


    # Write per-language
    for lang, splits in all_lang_splits.items():
        for split in ["train", "dev", "test"]:
            out_dir = os.path.join(output_data_dir, lang, split)

            write_kaldi_dir(splits[split], out_dir, dry_run=dry_run, base_dir=base_dir)
            output_dirs.append(out_dir)

    # Trilingual
    tri = {"train": [], "dev": [], "test": []}
    for lang in ["id", "ar", "en"]:
        for split in ["train", "dev", "test"]:
            tri[split].extend(all_lang_splits[lang][split])

    for split in ["train", "dev", "test"]:
        out_dir = os.path.join(output_data_dir, "tri", split)
        write_kaldi_dir(tri[split], out_dir, dry_run=dry_run, base_dir=base_dir)
        output_dirs.append(out_dir)

    if not dry_run and args.fix_data:
        for d in output_dirs:
            run_cmd(["./utils/fix_data_dir.sh", d])
            run_cmd(["./utils/validate_data_dir.sh", d, "--no-feats"])

    return output_dirs

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action='store_true')
    parser.add_argument("--use_mms", action='store_true', default=True,
                         help="If set (default: True), primary cs records "
                              "are folded into mms_output_data_dir (in addition "
                              "to data/cs being written normally). Pass "
                              "--no-use_mms to write cs_mms standalone only.")
    parser.add_argument("--no_use_mms", dest="use_mms", action='store_false',
                         help="Disable folding primary cs records into mms_output_data_dir.")
    parser.add_argument("--fix_data", action='store_true', default=False,
                         help="If set, run data validation, and data dir " 
                         "using default ESPnet scripts")
    parser.add_argument("--stage", type=int, default=1,
                         help="Stage to start from (default: 1).")
    parser.add_argument("--stop_stage", type=int, default=1000,
                         help="Last stage to run, inclusive (default: 1000, "
                              "i.e. run through the final stage).")
    args = parser.parse_args()

    base_dir = "downloads"
    manifest_dir = os.path.join(base_dir, "processed", "manifests", "balanced")

    base_output_data_dir = "data"
    mms_output_data_dir = os.path.join(base_output_data_dir, "cs_mms")        

    if (args.stage <= 1 <= args.stop_stage) and not args.dry_run:
        run_stage1(base_dir)

    if args.stage <= 2 <= args.stop_stage:
        run_stage2(base_dir)

    if args.stage <= 3 <= args.stop_stage:
        run_stage3(
            base_dir=base_dir,
            manifest_dir=manifest_dir,
            output_data_dir=base_output_data_dir,
            cs_mms_data_dir=mms_output_data_dir,
            use_mms=args.use_mms,
            dry_run=args.dry_run,
        )
