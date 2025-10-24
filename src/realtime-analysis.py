#!/usr/bin/env python3
"""
realtime_juggle_detector.py

Real-time juggling-audio detector that listens on the default microphone,
detects peaks (catches), tracks cycles for a given siteswap-like pattern,
and prints live feedback such as:
  - "Pattern forming..."
  - "✅ On rhythm"
  - "⚠️ Rushing by 80 ms"
  - "⚠️ Lagging by 40 ms"
  - "❌ Missed catch"

Usage:
    python realtime_juggle_detector.py --pattern 441
"""

import argparse
import queue
import time
import numpy as np
from scipy.signal import find_peaks
import sounddevice as sd
from collections import deque

# ---------------------------
# CONFIG (tweakable)
# ---------------------------
DEFAULT_SR = 44100
BLOCKSIZE = 2048                 # samples per callback (approx 46 ms @ 44.1kHz)
BUFFER_DURATION = 10.0           # seconds kept in rolling buffer
INITIAL_SILENCE_CAL = 3.0        # first N seconds used to estimate noise floor
PEAK_PARAMS = {
    "height": 0.002,             # minimal peak height (adjust for mic sensitivity)
    "distance_sec": 0.1,         # minimal time between peaks (seconds)
    "prominence": 0.004         # prominence threshold
}
TOLERANCE = 0.1                  # seconds tolerance for matching predictions
MISS_FACTOR = 1.5                # if gap > MISS_FACTOR * expected_interval -> consider missed

# ---------------------------
# Command-line args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Realtime juggling pattern detector")
    p.add_argument("--pattern", type=int, required=True, help="Siteswap-style pattern (e.g. 441, 51)")
    p.add_argument("--samplerate", type=int, default=DEFAULT_SR, help="Audio sample rate (Hz)")
    p.add_argument("--buffer", type=float, default=BUFFER_DURATION, help="Sliding buffer length (seconds)")
    p.add_argument("--silence", type=float, default=INITIAL_SILENCE_CAL,
                   help="Initial silence duration to calibrate noise floor (seconds)")
    return p.parse_args()

# ---------------------------
# Audio capture callback
# ---------------------------
audio_q = queue.Queue()

def audio_callback(indata, frames, timeinfo, status):
    if status:
        # print but don't flood
        print("Audio status:", status)
    # put copy of data (mono)
    audio_q.put(indata[:, 0].copy())

# ---------------------------
# Live-feedback logic
# ---------------------------
def estimate_expected_interval(catch_times, pattern_len):
    """Return expected interval (seconds) based on last pattern_len intervals, or None."""
    if len(catch_times) < pattern_len + 1:
        return None
    # use last pattern_len intervals
    intervals = np.diff(catch_times)
    last_intervals = intervals[-pattern_len:]
    # guard against NaN
    if np.any(np.isnan(last_intervals)) or len(last_intervals) == 0:
        return None
    return float(np.mean(last_intervals))

def classify_drift(actual_time, predicted_time):
    """Return drifting message and drift value (actual - predicted) in seconds."""
    if predicted_time is None:
        return "🎵 Pattern forming...", None
    drift = actual_time - predicted_time
    # small tolerance considered "on rhythm"
    if abs(drift) <= 0.05:
        return "✅ On rhythm", drift
    if drift < -0.05:
        # actual earlier than predicted -> rushing (negative drift)
        return f"⚠️ Rushing by {abs(drift)*1000:.0f} ms", drift
    else:
        return f"⚠️ Lagging by {abs(drift)*1000:.0f} ms", drift

# ---------------------------
# Main realtime detector
# ---------------------------
def main():
    args = parse_args()
    sr = args.samplerate
    buffer_dur = args.buffer
    silence_cal = args.silence
    pattern = args.pattern
    pattern_len = len(str(pattern))

    # Derived params
    buffer_samples = int(buffer_dur * sr)
    buf = np.zeros(buffer_samples, dtype=np.float32)

    distance_samples = int(PeakDistance := sr * PEAK_PARAMS["distance_sec"]) if False else int(sr * PEAK_PARAMS["distance_sec"])
    # track calibrations and state
    start_time = time.time()
    noise_floor = None
    calibrated = False
    recent_catches = deque()   # store absolute times (seconds since start_time) of detected peaks
    last_reported_peak_time = 0.0   # only announce peaks newer than this
    predicted_next = None  # predicted next catch time (seconds since start_time)
    last_actual_peak = None

    print("Starting real-time detector")
    print(f"Pattern: {pattern} (length {pattern_len}), buffer {buffer_dur}s, samplerate {sr} Hz")
    print("Calibrating noise floor for first {:.1f} seconds...".format(silence_cal))

    # start input stream
    try:
        with sd.InputStream(channels=1, samplerate=sr, blocksize=BLOCKSIZE, callback=audio_callback):
            print("🎤 Listening... (Ctrl+C to stop)\n")
            while True:
                # get next audio chunk (blocking)
                chunk = audio_q.get()
                if chunk is None:
                    continue

                # roll buffer and append
                n = len(chunk)
                buf = np.roll(buf, -n)
                buf[-n:] = chunk

                t_now = time.time() - start_time  # relative time

                # calibrate noise floor from first silence_cal seconds
                if not calibrated:
                    if t_now >= silence_cal:
                        # compute noise floor from initial tail of buffer corresponding to first silence_cal seconds
                        samples_for_cal = int(silence_cal * sr)
                        # since buffer may have wrapped, use last samples_for_cal of the buffer for calibration
                        noise_floor = np.mean(np.abs(buf[-samples_for_cal:]))
                        calibrated = True
                        print(f"Calibrated noise floor = {noise_floor:.6f}")
                    else:
                        # still calibrating; skip detection until calibrated
                        continue

                # Simple noise gating / normalization: subtract small constant noise_floor
                buf_clean = buf.copy()
                buf_clean = buf_clean - noise_floor
                # Ensure small values around zero
                # (clipping helpful to reduce negative residuals)
                buf_clean[np.abs(buf_clean) < noise_floor * 0.5] = 0.0

                # Detect peaks on the *entire* buffer but only report peaks newer than last_reported_peak_time
                peaks, props = find_peaks(
                    buf_clean,
                    height=PEAK_PARAMS["height"],
                    distance=int(sr * PEAK_PARAMS["distance_sec"]),
                    prominence=PEAK_PARAMS["prominence"]
                )

                # Convert peak indices to absolute times since start_time
                peak_times = (np.array(peaks, dtype=float) / sr) + (t_now - buffer_dur)
                # Filter to only new peaks
                new_peaks_mask = peak_times > last_reported_peak_time + 1e-9
                new_peak_times = peak_times[new_peaks_mask]

                # For stability, ignore any peaks in the "old half" of buffer (optional)
                # (already filtered by last_reported_peak_time but helps avoid duplicates)
                # Append new peaks to recent_catches
                for pt in new_peak_times:
                    recent_catches.append(pt)
                    last_reported_peak_time = max(last_reported_peak_time, pt)
                    last_actual_peak = pt
                    # Keep recent only (last buffer_dur seconds)
                    while recent_catches and (t_now - recent_catches[0] > buffer_dur):
                        recent_catches.popleft()

                    # For each newly detected actual catch, evaluate immediate feedback
                    expected_interval = estimate_expected_interval(list(recent_catches), pattern_len)
                    # If we have an expected interval, we can compute a predicted next based on the last
                    # cycle start predicted_next may be None until we set it
                    # If predicted_next exists (from previous prediction), compute drift; otherwise classify as forming
                    feedback_msg, drift = classify_drift(pt, predicted_next)

                    # update predicted_next based on current recent intervals (predict next actual)
                    if expected_interval is not None:
                        predicted_next = pt + expected_interval
                    else:
                        predicted_next = None

                    # Print a helpful message
                    # Also provide number of recent catches and short summary
                    short_info = f"(recent {len(recent_catches)} catches)"
                    print(f"{time.strftime('%H:%M:%S')}  {feedback_msg} {short_info}")

                # Periodic checks even if there are no new peaks:
                # e.g., detect missed catch if time since last_actual_peak > MISS_FACTOR * expected_interval
                # Only run this check every loop iteration; will print at most once per detected miss until next peak
                expected_interval_now = estimate_expected_interval(list(recent_catches), pattern_len)
                if expected_interval_now is not None and last_actual_peak is not None:
                    time_since_last = t_now - last_actual_peak
                    if time_since_last > expected_interval_now * MISS_FACTOR:
                        # Consider a missed catch (but don't spam: only announce if we haven't predicted a miss recently)
                        # We will mark predicted_next to avoid duplicate prints until next actual occurs
                        print(f"{time.strftime('%H:%M:%S')}  ❌ Missed catch (no catch in {time_since_last:.2f}s, expected ≈ {expected_interval_now:.2f}s)")
                        # nudging predicted_next forward to avoid repeated misses immediate spam
                        predicted_next = last_actual_peak + expected_interval_now

                # small sleep to avoid a busy loop (mainly when blocks are high-rate)
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    main()
