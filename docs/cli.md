# Command-Line Interface

nanodsp ships with a CLI accessible via `nanodsp` or `python -m nanodsp`.

## Process

Apply effect chains to audio files.

```bash
# Single file
nanodsp process input.wav -o output.wav \
  -f highpass:cutoff_hz=80 \
  -f compress:ratio=4,threshold=-18 \
  -f normalize_peak:target_db=-1

# Apply a preset
nanodsp process vocals.wav -o out.wav -p vocal_chain

# Combine preset with additional effects
nanodsp process input.wav -o out.wav -f lowpass:cutoff_hz=12000 -p master

# Batch mode
nanodsp process *.wav -O out/ -f lowpass:cutoff_hz=2000

# Dry run (show chain without processing)
nanodsp process input.wav -n -f highpass:cutoff_hz=80 -f compress:ratio=4
```

## Analyze

Measure audio properties.

```bash
nanodsp analyze input.wav loudness
nanodsp analyze input.wav pitch --fmin=80 --fmax=800
nanodsp analyze input.wav onsets --json
nanodsp analyze input.wav info
```

## Synthesize

Generate audio from scratch.

```bash
nanodsp synth tone.wav sine --freq=440 --duration=2.0
nanodsp synth kick.wav drum --type=analog_bass_drum --freq=60
nanodsp synth melody.wav note --instrument=clarinet --freq=440 --duration=1.0
nanodsp synth seq.wav sequence --instrument=flute \
  --notes='[{"freq":440,"start":0,"dur":0.5},{"freq":554,"start":0.5,"dur":0.5}]'
```

## Convert

Convert between formats and resample.

```bash
nanodsp convert input.wav output.flac
nanodsp convert input.wav output.wav --sample-rate=44100 --channels=1 -b 24
```

## Info

Display file metadata.

```bash
nanodsp info drums.wav
nanodsp info drums.wav --json
```

## Presets

30 built-in presets across 8 categories.

```bash
nanodsp preset list
nanodsp preset list spatial
nanodsp preset info master
nanodsp preset apply master input.wav output.wav target_lufs=-16
```

| Category | Presets |
|----------|---------|
| mastering | `master`, `master_pop`, `master_hiphop`, `master_classical`, `master_edm`, `master_podcast` |
| voice | `vocal_chain` |
| spatial | `room`, `hall`, `plate`, `cathedral`, `chamber` |
| dynamics | `gentle_compress`, `heavy_compress`, `brick_wall` |
| creative | `radio`, `underwater`, `megaphone`, `tape_warmth`, `shimmer`, `vaporwave`, `walkie_talkie` |
| lofi | `telephone`, `lo_fi`, `vinyl`, `8bit` |
| cleanup | `dc_remove`, `de_noise`, `normalize`, `normalize_lufs` |

## Pipe

stdin/stdout streaming for shell pipelines.

```bash
cat input.wav | nanodsp pipe -f lowpass:cutoff_hz=1000 > output.wav
cat input.wav | nanodsp pipe -p telephone > output.wav
nanodsp pipe -f highpass:cutoff_hz=80 < in.wav | nanodsp pipe -f compress:ratio=4 > out.wav
```

## Benchmark

Measure function throughput.

```bash
nanodsp benchmark lowpass:cutoff_hz=1000
nanodsp benchmark compress:ratio=4,threshold=-20 -n 100 --duration=2.0
nanodsp benchmark reverb:preset=hall --channels=2 --json
```

## List

Show available functions.

```bash
nanodsp list
nanodsp list filters
nanodsp list effects
```
