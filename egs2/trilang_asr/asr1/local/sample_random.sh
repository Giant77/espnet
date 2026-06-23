#!/usr/bin/env bash
set -euo pipefail

for split in train dev; do
    SRC_ID="data/id/$split"
    SRC_EN="data/en/$split"
    SRC_AR="data/ar/$split"
    SRC_CS="data/cs/$split"
    DEST="data/cs_tri/$split"

    mkdir -p "$DEST"

    extract_3_4_3() {
        local file="$1"
        local n mid start

        n=$(wc -l < "$file")

        (( n == 0 )) && return

        # Small files: just keep everything.
        if (( n <= 10 )); then
            cat "$file"
            return
        fi

        mid=$((n / 2))
        start=$((mid - 1))

        {
            seq 1 3
            seq "$start" "$((start + 3))"
            seq "$((n - 2))" "$n"
        } | sort -nu | while read -r i; do
            sed -n "${i}p" "$file"
        done
    }

    for f in "$SRC_CS"/*; do
        [ -f "$f" ] || continue

        name=$(basename "$f")
        out="$DEST/$name"

        : > "$out"

        [ -f "$SRC_ID/$name" ] && \
            extract_3_4_3 "$SRC_ID/$name" >> "$out"

        [ -f "$SRC_EN/$name" ] && \
            extract_3_4_3 "$SRC_EN/$name" >> "$out"

        [ -f "$SRC_AR/$name" ] && \
            extract_3_4_3 "$SRC_AR/$name" >> "$out"

        # Append the full CS file.
        cat "$SRC_CS/$name" >> "$out"
    done

    echo "Done: $split"
done