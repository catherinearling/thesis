import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from scipy.signal import find_peaks, medfilt



# ================================================================
# GLOBAL CONFIGURATION
# ================================================================
TOLERANCE = 0.06  # 60ms tolerance for matching predicted to actual throw

SILENCE_DURATION = 5  # first 5 seconds assumed to be no activity by default.

PEAK_DETECTION_PARAMS = {
    "height": 0.0005,  # minimum height of peaks. a value determined from data
    "distance_sec": 0.06,  # minimum time between peaks in seconds. value determined from data
    "prominence": 0.005 # how much a peak has to "stand out" relative to its surroundings
}

#for analyzing rhythm of detected catches
INTERVAL_BOUNDS = {
    "min": 0.06,  # minimum allowed interval (seconds)
    "max": 0.64   # maximum allowed interval (seconds)
}

def longest_common_subsequence(seq1, seq2, tolerance=TOLERANCE):
    m, n = len(seq1), len(seq2)

    # Create a 2D array to store lengths of LCS
    #   a 2D DP (dynamic programming) table 
    # where dp[i][j] 
    # represents the LCS length between 
    # the first i elements of seq1 
    # and the first j elements of seq2.
    dp = np.zeros((m + 1, n + 1))

    # Fill dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
                # if the two catches are within the tolerance, they are considered a match
                # so we extend the LCS found so far by 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                #otherwise, carry forward the max LCS found by 
                # either ignoring the current element of seq1 or seq2
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])


    return int(dp[m][n])


#------------------------------------------------------------
# Plot detected catches, predicted cycle starts, and predicted
# beat catches, with match lines
#------------------------------------------------------------
def plotCycles(catch_times, predicted_cycles, predicted_catches):
    plt.figure(figsize=(12, 5))

    offset = 0.02  # tiny bump for predicted points
    offset_catches = 0.04   # orange dots: predicted within-cycle catches 

    # Detetced catches (blue dots at 0)
    plt.scatter(catch_times, np.zeros_like(catch_times), label='Detected Catches', color='blue', marker='.')

    # Predicted cycles (red crosses slightly above)
    plt.scatter(predicted_cycles, np.full_like(predicted_cycles, offset), label='Predicted Cycle Starts', color='red', marker='.')

    # Predicted individual catches within cycles
    predicted_catches_arr = np.array(predicted_catches)
    plt.scatter(predicted_catches_arr, np.full_like(predicted_catches_arr, offset_catches),
                label='Predicted Beat Catches', color='orange', marker='|', s=60)


    # Highlight matches
    for predicted_time in predicted_cycles:
        close_matches = np.abs(catch_times - predicted_time) <= TOLERANCE
        if np.any(close_matches):
            actual_match_time = catch_times[close_matches][0]
            plt.plot([actual_match_time, predicted_time], [0, offset], color='green', linestyle='--', linewidth=1)

    # Highlight matches between predicted beat catches and detected catches
    for predicted_time in predicted_catches:
        close_matches = np.abs(catch_times - predicted_time) <= TOLERANCE
        if np.any(close_matches):
            actual_match_time = catch_times[close_matches][0]
            plt.plot([actual_match_time, predicted_time], [0, offset_catches],
                        color='purple', linestyle=':', linewidth=0.8)

    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.title('Estimated Cycle Starts/Ends vs Detected Catch Times')
    plt.yticks([])

    # FORCE a tight vertical range
    plt.ylim(-0.05, 0.07)   # <<< prevents huge empty space

    #plt.show()


#returns nearest catch to current_time in catch_times array. also returns the index of this catch
def nearestCatch(catch_times, current_time):
    index = np.argmin(np.abs(catch_times - current_time))
    return catch_times[index], index

def plotPeaksComparison(time, time_clean, audio, audio_clean, sampling_rate, original_peaks, clean_peaks):
    #Visualization--------------------------------------------------------------------
    # Visualize both graphs with detected peaks side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Two graphs side by side

    # Cleaned audio
    ax[0].plot(time_clean, audio_clean)
    ax[0].plot(time_clean[clean_peaks], audio_clean[clean_peaks], "x", color='red')
    ax[0].set_title("Detected Peaks (Cleaned & Trimmed Audio)")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Amplitude")

    # Original audio
    ax[1].plot(time, audio)
    ax[1].plot(time[original_peaks], audio[original_peaks], "x", color='red')
    ax[1].set_title("Detected Peaks (Original Audio)")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def pdf(x, mu=0.0, sigma=1.0):
    """
    Gaussian (normal) probability density function.
      x: value to evaluate
      mu: mean
      sigma: standard deviation
    """
    coeff = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    exponent = math.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))
    return coeff * exponent



def parse_args():
    parser = argparse.ArgumentParser(description="Analyze rhythmic patterns in juggling audio.")
    parser.add_argument("--file", required=True, help="Name of audio file located in the /data folder. First 5 seconds should be no activity.")
    parser.add_argument("--pattern", type=int, required=False, help="Vanilla siteswap pattern (like 441, 51)")
    parser.add_argument("--silence", type=int, required=False, help="Duration of initial silence in seconds (default: 5)",default=SILENCE_DURATION)
    return parser.parse_args()

def detect_peaks_dynamic(audio_clean, sampling_rate,
                         min_distance_sec,
                         height_percentile=99,
                         prominence_percentile=99):
    x = np.asarray(audio_clean)

    # abs so we catch both positive & negative spikes
    abs_x = np.abs(x)

    #  dynamic thresholds
    # finds the value below which a given percentage of data points fall
    height_thr = np.percentile(abs_x, height_percentile)
    # only peaks with prominence above the specified percentile are considered significant
    prom_thr   = np.percentile(abs_x, prominence_percentile)

    # Enforce a minimum distance in time
    min_distance = int(sampling_rate * min_distance_sec)

    peaks, properties = find_peaks(
        x,
        height=height_thr,
        distance=min_distance,
        prominence=prom_thr
    )
    return peaks, properties

