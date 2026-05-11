# Juggling Audio Analysis

A Python tool that analyzes audio recordings of juggling to detect catch times, identify rhythmic patterns, and provide coaching feedback on timing consistency.

## What It Does

- Loads a `.wav` recording of juggling
- Reduces background noise using Short-Time Fourier Transform (STFT)
- Detects audio peaks corresponding to catch/throw sounds
- Estimates cycle timing based on a known or auto-detected siteswap pattern
- Outputs coaching tips on timing consistency, tempo drift, and phase slips
- Exports detected peak timestamps and predicted cycle start times to `.txt` files
- Visualizes detected vs. predicted catches in a plot

## Project Structure

```
audio/
├── data/               # Audio input files and output .txt files go here
├── src/
│   ├── jugglingAnalysis.py   # Main entry point
│   └── analysisHelpers.py    # Shared utilities (peak detection, plotting, etc.)
```

## Setup

```bash
pip install librosa soundfile scipy numpy matplotlib
```

## Usage

```bash
python .\src\jugglingAnalysis.py --file "yourfile.wav" --pattern 441
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--file` | Yes | Name of the `.wav` file in the `data/` folder |
| `--pattern` | No | Vanilla siteswap pattern (e.g. `3`, `441`, `51`). If omitted, the tool attempts to auto-detect it. |
| `--silence` | No | Duration of silence at the start of the recording in seconds (default: 5) |

**Example:**

```bash
python .\src\jugglingAnalysis.py --file "juggling_441.wav" --pattern 441 --silence 3
```

## Output

- `data/<filename>.txt` — timestamps (in seconds) of every detected catch peak
- `data/<filename>-pattern.txt` — timestamps of predicted cycle starts
- Console output with coaching tips and an overall pattern match score
- Two plots: raw vs. cleaned audio with detected peaks, and cycle alignment visualization

## Notes

- The audio file must be placed in the `data/` folder before running.
- The first N seconds of the recording are assumed to be silence/no activity and are used for noise estimation. Use `--silence` to set this duration.
- Peak detection parameters (height, prominence, minimum distance) can be tuned in the `PEAK_DETECTION_PARAMS` dictionary at the top of `analysisHelpers.py`.
