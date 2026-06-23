#!/bin/bash
set -euo pipefail

PERCENT=10

SRC_ID="data/id/train"
SRC_EN="data/en/train"
SRC_AR="data/ar/train"
SRC_CS="data/cs/train"
DEST="data/cs_tri/train"

mkdir -p "$DEST"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

make_index() {
    local dir="$1"
    local outfile="$2"

    local ref
    ref=$(find "$dir" -type f | head -n1)

    [ -n "$ref" ] || {
        echo "No files in $dir"
        exit 1
    }

    local total take
    total=$(wc -l < "$ref")
    take=$((total * PERCENT / 100))

    shuf -i 1-"$total" -n "$take" | sort -n > "$outfile"
}

# One random index per folder
make_index "$SRC_ID" "$tmpdir/id.idx"
make_index "$SRC_EN" "$tmpdir/en.idx"
make_index "$SRC_AR" "$tmpdir/ar.idx"

for f in "$SRC_CS"/*; do
    name=$(basename "$f")
    out="$DEST/$name"

    : > "$out"

    # Random lines from id
    if [ -f "$SRC_ID/$name" ]; then
        awk 'NR==FNR {a[$1]; next} FNR in a' \
            "$tmpdir/id.idx" "$SRC_ID/$name" >> "$out"
    fi

    # Random lines from en
    if [ -f "$SRC_EN/$name" ]; then
        awk 'NR==FNR {a[$1]; next} FNR in a' \
            "$tmpdir/en.idx" "$SRC_EN/$name" >> "$out"
    fi

    # Random lines from ar
    if [ -f "$SRC_AR/$name" ]; then
        awk 'NR==FNR {a[$1]; next} FNR in a' \
            "$tmpdir/ar.idx" "$SRC_AR/$name" >> "$out"
    fi

    # Entire cs file
    cat "$SRC_CS/$name" >> "$out"
done

echo "Done."
