#!/usr/bin/env bash
#
# Play demo renders by keyword, using `play` from sox.
#
# `make demos` writes a few hundred .wav files into build/demo-output/, named
# <source>_<demo>_<label>.wav. That is far too many to audition by hand, and
# the interesting comparisons are between files sharing part of a name -- one
# effect across its parameter sweep, or one parameter across several effects.
# This filters by substring and plays what matches, in order.
#
#   ./demos/play.sh keyframe splice     # every splice-threshold render
#   ./demos/play.sh -l reverb           # list, do not play
#   ./demos/play.sh -r 'stretch-(2|4)x' # regex instead of substring
#
# Multiple keywords narrow rather than widen: a file must match all of them.

set -uo pipefail

DIR="build/demo-output"
LIST_ONLY=0
REGEX=0
LIMIT=0
GAIN=""

usage() {
    cat <<'EOF'
Usage: demos/play.sh [options] KEYWORD [KEYWORD ...]

Plays .wav files from the demo output directory whose names match every
KEYWORD given. Matching is case-insensitive substring by default.

Options:
  -d DIR    Directory to search (default: build/demo-output)
  -l        List matches and exit without playing
  -r        Treat keywords as extended regular expressions
  -n N      Play at most N files
  -g GAIN   Volume multiplier passed to play -v (e.g. 0.5 to halve)
  -h        Show this help

Examples:
  demos/play.sh keyframe               # every keyframe render
  demos/play.sh keyframe sparsify      # only the sparsify sweep
  demos/play.sh -l compare             # list the three-way comparisons
  demos/play.sh -r 'pitch-(up|down)'   # regex
  demos/play.sh -n 5 -g 0.5 reverb     # first five, at half volume

Exit status is 1 if nothing matched or a file failed to play.
EOF
}

while getopts ":d:lrn:g:h" opt; do
    case "$opt" in
        d) DIR="$OPTARG" ;;
        l) LIST_ONLY=1 ;;
        r) REGEX=1 ;;
        n) LIMIT="$OPTARG" ;;
        g) GAIN="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) echo "play.sh: option -$OPTARG requires an argument" >&2; exit 2 ;;
        \?) echo "play.sh: unknown option -$OPTARG" >&2; usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ "$#" -eq 0 ]; then
    echo "play.sh: no keyword given" >&2
    usage >&2
    exit 2
fi

if [ "$LIST_ONLY" -eq 0 ] && ! command -v play >/dev/null 2>&1; then
    echo "play.sh: 'play' not found -- install sox (apt install sox, brew install sox)" >&2
    exit 1
fi

if [ ! -d "$DIR" ]; then
    echo "play.sh: $DIR does not exist -- run 'make demos' first" >&2
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "play.sh: -n takes a non-negative integer, got '$LIMIT'" >&2
    exit 2
fi

# Collect matches. A file has to satisfy every keyword, so the common case --
# narrowing a broad match down -- works by adding words rather than crafting a
# pattern.
shopt -s nullglob nocasematch
matches=()
for path in "$DIR"/*.wav; do
    name=$(basename "$path")
    keep=1
    for kw in "$@"; do
        if [ "$REGEX" -eq 1 ]; then
            [[ "$name" =~ $kw ]] || { keep=0; break; }
        else
            [[ "$name" == *"$kw"* ]] || { keep=0; break; }
        fi
    done
    [ "$keep" -eq 1 ] && matches+=("$path")
done
shopt -u nocasematch

if [ "${#matches[@]}" -eq 0 ]; then
    echo "play.sh: nothing in $DIR matches: $*" >&2
    exit 1
fi

# Deterministic order. Version sort where available, so a parameter sweep plays
# as k4, k16, k64, k256 rather than lexicographically as k16, k256, k4, k64.
# BSD sort gained -V late, so fall back rather than assume it.
if printf 'a2\na10\n' | sort -V >/dev/null 2>&1; then
    sort_cmd=(sort -V)
else
    sort_cmd=(sort)
fi
IFS=$'\n' matches=($(printf '%s\n' "${matches[@]}" | "${sort_cmd[@]}")) ; unset IFS

if [ "$LIMIT" -gt 0 ] && [ "${#matches[@]}" -gt "$LIMIT" ]; then
    matches=("${matches[@]:0:$LIMIT}")
    limited=" (limited to $LIMIT)"
else
    limited=""
fi

total=${#matches[@]}

duration() {
    if command -v soxi >/dev/null 2>&1; then
        soxi -D "$1" 2>/dev/null | awk '{printf "%6.2fs", $1}'
    else
        printf "       "
    fi
}

if [ "$LIST_ONLY" -eq 1 ]; then
    echo "$total file(s) match: $*$limited"
    for path in "${matches[@]}"; do
        printf '  %s  %s\n' "$(duration "$path")" "$(basename "$path")"
    done
    exit 0
fi

# Ctrl-C should stop the run, not just skip to the next file. `play` traps
# SIGINT itself and exits cleanly, which would otherwise leave the loop going.
interrupted=0
trap 'interrupted=1' INT

echo "Playing $total file(s) matching: $*$limited"
play_args=(-q)
[ -n "$GAIN" ] && play_args+=(-v "$GAIN")

failed=0
index=0
for path in "${matches[@]}"; do
    index=$((index + 1))
    printf '[%*d/%d] %s  %s\n' "${#total}" "$index" "$total" \
        "$(duration "$path")" "$(basename "$path")"
    play "${play_args[@]}" "$path" 2>/dev/null || failed=$((failed + 1))
    if [ "$interrupted" -eq 1 ]; then
        echo "Interrupted after $index of $total."
        exit 130
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "$failed of $total failed to play (no audio device?)" >&2
    exit 1
fi
