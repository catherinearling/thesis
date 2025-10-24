import sys
import os #for file path handling
import argparse

from scipy.signal import find_peaks, medfilt
import numpy as np
import matplotlib.pyplot as plt

import librosa
import soundfile as sf


# ================================================================
# GLOBAL CONFIGURATION
# ================================================================
TOLERANCE = 0.1  # 100ms tolerance for matching predicted to actual throw
SILENCE_DURATION = 5  # first 5 seconds assumed to be no activity

PEAK_DETECTION_PARAMS = {
    "height": 0.001,  # minimum height of peaks
    "distance_sec": 0.1,  # minimum time between peaks in seconds
    "prominence": 0.004 # how much a peak has to "stand out" relative to its surroundings
}

INTERVAL_BOUNDS = {
    "min": 0.1,  # minimum allowed interval (seconds)
    "max": 0.4   # maximum allowed interval (seconds)
}


def longest_common_subsequence(seq1, seq2, tolerance=TOLERANCE):
    m, n = len(seq1), len(seq2)

    # Create a 2D array to store lengths of LCS
    #   a 2D DP (dynamic programming) table 
    # where dp[i][j] 
    # represents the LCS length between 
    # the first i elements of seq1 
    # and the first j elements of seq2.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

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

    #Backtrack from bottom-right to recontrsuct the LCS
    lcs_matches = []
    i, j = m, n
    while i > 0 and j > 0:
        if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
            lcs_matches.append((seq1[i - 1], seq2[j - 1])) # store both so we can see how close they were
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return int(dp[m][n]), lcs_matches[::-1]  # return length and the LCS itself (reversed to correct order)


#------------------------------------------------------------
# Analyze the timing intervals between detected peaks to estimate rhythmic juggling cycles
# ----assumes the times are in order 
#  paramters:
# --peaks: indices of detected catch times / spikes
# --time: time values corresponding to audio samples
# --pattern: the siteswap pattern (441, 3, 531, etc)
#------------------------------------------------------------
def analyzeIntervals(peaks, time, pattern):
    pattern_length = len(str(pattern))
    #put intervals btwn catches into data struct

    #convert peak indices to times
    catch_times = time[peaks]
    print(f"Detected {len(catch_times)} catch times.")

    if len(catch_times) == 0:
        print("No peaks detected. Try adjusting noise reduction or peak detection params.")
        return
    if len(catch_times) < pattern_length:
        print(f"Not enough catches detected for pattern {pattern}. Either more data is needed for analyzation, \
              or try adjusting noise reduction or peak detection params.")
        return

    # Expected reasonable min and max interval
    min_interval = INTERVAL_BOUNDS["min"]
    max_interval = INTERVAL_BOUNDS["max"]

    # Identify catches outside expected bounds
    #irregular_intervals = (intervals < min_interval) | (intervals > max_interval)

    # ---------Estimate cycle length
    # every ball must be thrown and caught once per cycle
    # Rough estimate of cycle time based on initial catches
    # this is considering every single spike as a catch time!!!
    first_catch_time = catch_times[0]

    accuracy = 0
    predicted_cycles = []
    predictions = 0
    matches = 0
    drift_variability = 0
    drift = 0
    lcs = []

    # Use multiple guesses for cycle durations based on different starting catch pairs
    #consider each possible cycle length btwn ith catch and first catch -- start from i = pattern
    for i in range(pattern_length, len(catch_times)):
        cycle_duration_guess = catch_times[i] - first_catch_time

        #if guess is within reasonable estimates, consider it
        if(cycle_duration_guess <= (max_interval * pattern_length)) \
            and (cycle_duration_guess >= (min_interval * pattern_length)):
            # Predict where future cycles should occur based on this first cycle length
            # Start at first detected cycle
            predicted_cycle_starts = []
            current_time = first_catch_time
            total_matches = 0

            # Predict cycle start times by stepping forward a cycle length in time
            while current_time <= catch_times[-1]: #this would end too early if we are missing peaks at end 
                predicted_cycle_starts.append(current_time)
                
                # Find the index of the closest actual catch time to the current time
                index = np.argmin(np.abs(catch_times - current_time))
                closest_catch_time = catch_times[index]
                diff = np.abs(closest_catch_time - current_time)

                # adjust how much we advance by if our next predicted cycle start is close to an actual catch time
                if diff <= TOLERANCE:
                    total_matches += 1
                    current_time = closest_catch_time + cycle_duration_guess
                else:
                    # otherwise, just advance by our cycle duration guess to try to find next catch
                    current_time += cycle_duration_guess

            # ACCURACY SCORING ------
            #make accuracy score based off off LCS
            lcs_length, lcs_matches = longest_common_subsequence(predicted_cycle_starts, catch_times, tolerance=TOLERANCE)

            total_predictions = len(predicted_cycle_starts)
            precision = lcs_length / total_predictions if total_predictions > 0 else 0
            expected_cycles = len(catch_times) / pattern_length

            # Penalize overprediction more than underprediction
                #underpredicting is okay because we may have missed some catches because they were too quiet,
                # or we may have detected too many peaks due to noise
            ratio = total_predictions / expected_cycles
            # when ratio > 1, we are overpredicting
            if ratio > 1:
                #e^x  function gives stronger penalty for overpredicting number of cycles
                penalty = np.exp(1 - ratio) 
            # when ratio < 1, we are underpredicting
            elif ratio < 1:
                penalty = 0.85 + (0.25 * ratio)  # soft penalty if underpredicting

            #scale by penalty 
            curr_accuracy = precision * penalty
            #truncate
                #why? because sometimes floating point errors can cause it to be slightly > 1,
                    # or because, if the intervals between throws is really small, maybe 
                    # we can have multiple cycles match to snigle detected peak
            curr_accuracy = min(curr_accuracy, 1.0) 


            avg_drift = 0
            drift_var = 0
            if lcs_matches:
                # Compute global drift
                drift_values = []

                for pred_time in predicted_cycle_starts:
                    # Find closest actual catch
                    index = np.argmin(np.abs(catch_times - pred_time))
                    closest_catch_time = catch_times[index]
                    drift = closest_catch_time - pred_time   # positive means the actual catch is later
                    drift_values.append(drift)

                # Convert to numpy array for convenience
                drift_values = np.array(drift_values)

                # --- Global drift: average offset (mean of all differences)
                avg_drift = np.mean(drift_values)

                # --- Drift variability: how much drift fluctuates over time
                    #could use to see if performer is speeding up or slowing down
                drift_var = np.std(drift_values)

            #keep this prediction if it has best accuracy so far, and is a valid accuracy (0 < acc < 1)
                #valid accuracy means we are not overpredicting the amount of cycles nor underpredicting
            if curr_accuracy > 0 and curr_accuracy < 1 and curr_accuracy > accuracy:
                predicted_cycles = predicted_cycle_starts
                accuracy = curr_accuracy
                predictions = total_predictions
                matches = total_matches
                lcs = lcs_matches
                drift = avg_drift
                drift_variability = drift_var
        #if cycle is outside of reasonable bounds for time length, break out of for loop
        else:
            continue
    
    #only plot the best (in terms of accuracy) predictions we got
    if accuracy > 0 and len(predicted_cycles) > 0:
        print(f"Total predicted cycle starts: {predictions}")
        print(f"Matched cycle starts to detected catches: {matches}")
        print(f"Accuracy: {accuracy*100:.2f}%\n")
        print(f"Average drift between predicted and actual catch times: {drift*1000:.2f} ms")

        # Plotting
        plt.figure(figsize=(12, 5))
        offset = 0.02  # tiny bump for predicted points

        # Detetced catches (blue dots at 0)
        plt.scatter(catch_times, np.zeros_like(catch_times), label='Detected Catches', color='blue', marker='.')

        # Predicted cycles (red crosses slightly above)
        plt.scatter(predicted_cycles, np.full_like(predicted_cycles, offset), label='Predicted Cycle Starts', color='red', marker='.')

        # Highlight matches
        for predicted_time in predicted_cycles:
            close_matches = np.abs(catch_times - predicted_time) <= TOLERANCE
            if np.any(close_matches):
                actual_match_time = catch_times[close_matches][0]
                plt.plot([actual_match_time, predicted_time], [0, offset], color='green', linestyle='--', linewidth=1)
        
        if lcs:
            offsets = np.array([a-b for a,b in lcs])
            times = np.array([a for a,b in lcs])
            threshold = INTERVAL_BOUNDS["max"]

            for i, drift_val in enumerate(offsets):
                if abs(drift_val) > threshold:
                    color = 'red' if drift_val < 0 else 'blue'  # red = rushing, blue = lagging
                    plt.axvspan(times[i] - 0.1, times[i] + 0.1, color=color, alpha=0.15)

            plt.text(times[-1], 0.06, "Red = rushing | Blue = lagging", fontsize=9, ha='right')


        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.title('Detected Catch Times vs Predicted Cycle Starts (with drift)')
        plt.yticks([])

        # FORCE a tight vertical range
        plt.ylim(-0.05, 0.07)   # <<< prevents huge empty space
        
        plt.tight_layout()
        plt.show()
    
    else:
        print("could not find any valid cycle guesses")

#-----------------------------------------------------------------
# Apply Short Time Fourier Transform for background noise reduction
#       --breaks audio into small time windows and analyzes their frequencies
#       --assumes that the first {silence_duration} seconds of given audio are no activity / "silence"
#-----------------------------------------------------------------
def reduceNoise(audio, sampling_rate, silence_duration=SILENCE_DURATION):
    # S_full is the magnitude of each frequency at each time step; phase holds phase information for reconstructing audio later
    S_full, phase = librosa.magphase(librosa.stft(audio)) #this separates mag and phase data

    # Estimate background noise by calculating the mean power of the initial given seconds
    # This assumes the first given seconds are "silence" (low noise), hence a good reference for noise power
    noise_power = np.mean(S_full[:, :int(sampling_rate * silence_duration)], axis=1) #axis=1 ensures mean is calc for each freq band

    # Create a binary mask that identifies significant audio regions (above noise power threshold)
    #      a mask is a filter that IDs important data points
    #      any val in S_full > corresponding noise power is true (keep it)
    mask = S_full > noise_power[:, None]

    # Convert boolean mask to float values (1 for True, 0 for False)
    mask = mask.astype(float)

    # median filter reduces noisy data by averaging values in a sliding window
    # kernel_size=(1, 5) creates a filter that looks at 5 consecutive data points 
    # for each frequency band
    #reduce random background noise spikes while preserving peaks
    mask = medfilt(mask, kernel_size=(1,5))

    # apply mask to original data
    S_clean = S_full * mask

    # Inverse STFT to reconstruct cleaned audio by combining the filtered magnitude (S_clean) 
    # with the original phase data (phase)
    audio_clean = librosa.istft(S_clean * phase)

    return audio_clean


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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze rhythmic patterns in juggling audio.")
    parser.add_argument("--file", required=True, help="Name of audio file located in the /data folder. First 5 seconds should be no activity.")
    parser.add_argument("--pattern", type=int, required=True, help="Vanilla siteswap pattern(441, 51, etc)")
    parser.add_argument("--silence", type=int, required=False, help="Duration of initial silence in seconds (default: 5)",default=SILENCE_DURATION)
    return parser.parse_args()


# how to run from cmd line:
#   python <path/to/fileName> --file <audio-file-in-/data> --pattern <siteswap-pattern>
# e.g.
#   python src/jugglingAnalysis-experiment.py --file test1.wav --pattern 441
def main():
    args = parse_args()
    filename = args.file
    pattern = args.pattern
    silence_duration = args.silence

    # Load audio file -- the first 5 seconds are no juggling
    audio_path = os.path.join(os.getcwd(), "data", filename)

    if not os.path.exists(audio_path):
        print(f"File '{filename}' not found in /data.")
        return

    #sr=None preserves file's original sampling rate
    audio, sampling_rate = librosa.load(audio_path, sr=None)
    
    #background noise reduction
    audio_clean = reduceNoise(audio, sampling_rate, silence_duration)

    #remove any silence at the beginning (specified by user, default is 5 seconds)
    num_samples = int(sampling_rate * silence_duration)
    audio_clean = audio_clean[num_samples:]

    # Save the cleaned audio to a new file for debugging
    sf.write('clean.wav', audio_clean, sampling_rate)

    #---------------------------------------------------------------------------

    # Convert sample indices to time values for x-axis 
    time_clean = (np.arange(len(audio_clean)) + num_samples) / sampling_rate
    time = np.arange(len(audio)) / (sampling_rate)


    # Detect peaks (using time-based distance) for cleaned vs not audio. using these parameter values
    # based on results with test data
    # - height ensures peaks have a minimum amplitude
    # - distance=int(samplingRate * 0.1) ensures peaks are at least 0.1 seconds apart
    # - prominence ensures peaks must stand out by at least "prominence" relative to their surroundings
    clean_peaks, properties = find_peaks(
        audio_clean,
        height= PEAK_DETECTION_PARAMS["height"],
        distance=int(sampling_rate * PEAK_DETECTION_PARAMS["distance_sec"]),
        prominence= PEAK_DETECTION_PARAMS["prominence"]
    )

    original_peaks, properties = find_peaks(
        audio, 
        height= PEAK_DETECTION_PARAMS["height"], 
        distance=int(sampling_rate * PEAK_DETECTION_PARAMS["distance_sec"]), 
        prominence= PEAK_DETECTION_PARAMS["prominence"]
    )

    #plot and analyze intervals
    analyzeIntervals(clean_peaks, time_clean, pattern)

    plotPeaksComparison(time, time_clean, audio, audio_clean, sampling_rate, original_peaks, clean_peaks)

if __name__ == "__main__":
    main()