import sys
import os #for file path handling
import argparse
import math

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
    num_silence_frames =int(sampling_rate * silence_duration)
    noise_region = S_full[:, :num_silence_frames] 
    noise_power = np.mean(noise_region, axis=1) #axis=1 ensures mean is calc for each freq band


    noise_mean = np.mean(noise_region, axis=1)
    noise_std  = np.std(noise_region, axis=1)
    # threshold is mean + k * std, so only clearly-above-noise stuff survives
    noise_margin_std = 1.2  # how much above the mean noise power to set threshold
    noise_threshold = noise_mean + noise_margin_std * noise_std


    # Create a binary mask that identifies significant audio regions (above noise power threshold)
    #      a mask is a filter that IDs important data points
    #      any val in S_full > corresponding noise power is true (keep it)
    mask = S_full > noise_power[:, None]
    #mask = S_full > noise_threshold[:, None]

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

#------------------------------------------------------------
# Analyze the timing intervals between detected peaks to estimate rhythmic juggling cycles
# ----assumes the times are in order 
#  paramters:
# --peaks: indices of detected catch times / spikes
# --time: time values corresponding to audio samples
# --pattern: vanilla siteswap pattern (like 441, 51)
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
        print(f"Not enough catches detected for pattern length {pattern_length}. Either more data is needed for analyzation, \
              or try adjusting noise reduction or peak detection params.")
        return

    # Expected reasonable min and max interval .1 seconds, max interval .5 seconds
    min_interval = INTERVAL_BOUNDS["min"]
    max_interval = INTERVAL_BOUNDS["max"]

    # Identify catches outside expected bounds
    #irregular_intervals = (intervals < min_interval) | (intervals > max_interval)

    # ---------Estimate cycle length
    # every ball must be thrown and caught once per cycle
    # Rough estimate of cycle time based on initial catches
    # this is considering every single spike as a catch time!!!
    first_catch_time = catch_times[0]

    predicted_cycles = []
    predictions = 0
    num_matches = 0
    best_avg_log_prob = 0 # Initialize to first guess inside loop
    initialized = False
    best_sigma = None

    # Use multiple guesses for cycle durations based on different starting catch pairs
    #consider each possible cycle length btwn ith catch and first catch -- start from i = pattern_length
    for i in range(pattern_length, len(catch_times)):

        variation = 0
        begin = i - pattern_length #starts from 0, goes to len - pattern_length

        # --------- Cycle Duration Guessing and Matching ---------
        # slide a window of different cycle lengths over the array of catch times
        # this allows us to consider small errors in catch detection timing, since if we only try
        # one fixed cycle length over whole array, any small error in catch detection timing 
        # could throw off the entire analysis
        end = i + variation 
        while end < len(catch_times) and variation <= 2:  
            # Guess cycle duration with sliding window of cycle durations
            cycle_duration_guess = catch_times[end] - catch_times[begin]
            
            #if guess is within reasonable estimates, consider i
            if(cycle_duration_guess <= (max_interval * pattern_length)) \
                and (cycle_duration_guess >= (min_interval * pattern_length)):

                predicted_cycle_starts = []
                predicted_cycle_starts.append(catch_times[end])
                predicted_cycle_starts.append(catch_times[begin])
                
                current_time = first_catch_time
                curr_num_matches = 0

                #step forwards starting at end to find backwards matches for this cycle duration guess
                current_time = catch_times[end]
                while (current_time + cycle_duration_guess) <= catch_times[-1]:
                    current_time += cycle_duration_guess
                    predicted_cycle_starts.append(current_time)

                    # # If we step one cycle length away from our current time, 
                    # # do we land on a detected catch? (within tolerance)
                    # closest = nearestCatch(catch_times, current_time)
                    
                    # # i think maybe we should get rid of this tolerance check and just use the pdf scoring later
                    # # TODO ------------------------------------------------------
                    # diff = np.abs(closest - current_time)
                    # if diff <= TOLERANCE:
                    #     curr_num_matches += 1
                    #     current_time = closest #move to actual catch time, so that next step is relative to that

                #step backwards starting at index i-pattern_length to find forwards matches for this cycle duration guess
                if begin >= 0:
                    current_time = catch_times[begin]
                    while (current_time - cycle_duration_guess) >= first_catch_time:
                        current_time -= cycle_duration_guess
                        predicted_cycle_starts.insert(0, current_time)

                        # # If we step one cycle length away from our current time, 
                        # # do we land on a detected catch? (within tolerance)
                        # closest = nearestCatch(catch_times, current_time)

                        # # diff = np.abs(closest - current_time)
                        # if diff <= TOLERANCE:
                        #     curr_num_matches += 1
                        #     current_time = closest #move to actual catch time, so that next step is relative to that


                # --------- Scoring ---------
                # A candidate cycle is “good” if its predicted starts sit consistently 
                # close to real catch times
                
                # Choose sigma (the expected jitter around ideal times)
                best_sigma = TOLERANCE*0.85 #less strict for bad matches if its bigger

                # Compute product of Gaussian PDFs for how close each predicted time is to an actual catch
                log_prob_sum = 0.0
                for predicted_time in predicted_cycle_starts:
                    nearest = nearestCatch(catch_times, predicted_time)
                    diff = nearest - predicted_time
                    p = pdf(diff, mu=0, sigma=best_sigma)

                    #avoid multiplying many small probabilities together (which can lead to numerical underflow)
                    # instead, sum the log probabilities
                    log_prob_sum += math.log(p)
                
                avg_log_prob = log_prob_sum / len(predicted_cycle_starts)

                # Initialize best guess on first valid prediction
                #otherwise, keep this prediction if it has best accuracy so far, and is a valid accuracy (0 < acc < 1)
                    #valid accuracy means we are not overpredicting the amount of cycles nor underpredicting
                if (avg_log_prob > best_avg_log_prob) or (not initialized):
                    if not initialized:
                        initialized = True
                    predicted_cycles = predicted_cycle_starts
                    best_avg_log_prob = avg_log_prob
                    predictions = len(predicted_cycle_starts)
                    num_matches = curr_num_matches
            
            variation += 1  
            end = i + variation

    #only plot the best (in terms of accuracy) predictions we got
    if len(predicted_cycles) > 0 and best_sigma is not None:
        print(f"Total predicted cycle starts: {predictions}")
        print(f"Matched cycle starts to detected catches: {num_matches}")
        
        # normalize accuracy so it's roughly 0 to 100%
        accuracy = math.exp(best_avg_log_prob) / pdf(0, 0, best_sigma) * 100
        print(f"Pattern match score (accuracy): {accuracy}%\n")


        # Coaching Tips ------------------------------------------------
        intervals = np.diff(catch_times)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        estimated_cycles = len(catch_times) / pattern_length

        print("Coaching Tips:")
        variability = std_interval / avg_interval
        print(f"variability (std dev / mean): {variability:.2f}")
        if variability > 0.4:
            print("- Your timing between throws seems inconsistent." \
            " Some throws are much higher or lower than others. Try to make each throw identical in height.")
        elif variability > 0.3:
            print("Your rhythm is pretty good, but there's some room for improvement." \
            " Focus on maintaining a steady pace throughout your juggling.")  
        else:
            print("Great job! Your juggling rhythm is very consistent.")  

        
        # 3) Tempo drift: speeding up or slowing down over time
        if len(intervals) >= 6:
            third = len(intervals) // 3
            early_mean = np.mean(intervals[:third])
            late_mean = np.mean(intervals[-third:])
            if late_mean > early_mean * 1.3:
                print("You’re slowing down as you go. Try to keep the same tempo from start to finish.")
            elif late_mean < early_mean * 0.9:
                print("You’re speeding up over time. Try to keep your throws relaxed and steady.")    

        # 4) Pattern match quality
        if accuracy >= 80:
            print("Your timing matches the chosen pattern very well. Great job!")
        elif accuracy >= 50:
            print("You’re roughly following the pattern, but there are some off-beat cycles. "
                            "Focus on keeping the throws in a steady groove.")
        else:
            print("The timing doesn’t strongly match a repeating pattern yet. "
                            "Try working on a slower, simpler pattern and aim for very even throws.")

        # 5) Does the detected pattern cover most of the run?
        if estimated_cycles > 0:
            coverage_ratio = predictions / estimated_cycles
            if coverage_ratio < 0.6:
                print("The detected pattern only lines up with part of the run. "
                                "This could mean extra out-of-rhythm throws or drops between clean cycles.")

        # Plotting ------------------------------------------------
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

        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.title('Estimated Cycle Starts/Ends vs Detected Catch Times')
        plt.yticks([])

        # FORCE a tight vertical range
        plt.ylim(-0.05, 0.07)   # <<< prevents huge empty space

        plt.show()
    else:
        print("could not find any valid cycle guesses")


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


def nearestCatch(catch_times, current_time):
    index = np.argmin(np.abs(catch_times - current_time))
    return catch_times[index]

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
    parser.add_argument("--pattern", type=int, required=True, help="Vanilla siteswap pattern (like 441, 51)")
    parser.add_argument("--silence", type=int, required=False, help="Duration of initial silence in seconds (default: 5)",default=SILENCE_DURATION)
    return parser.parse_args()

def detect_peaks_dynamic(audio_clean, sampling_rate,
                         min_distance_sec=0.1,
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


# how to run from cmd line:
#   python .\src\jugglingAnalysis.py --file "hamerly_juggling_441.wav" --pattern 441
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
    sf.write('data/clean.wav', audio_clean, sampling_rate)

    #---------------------------------------------------------------------------
    # Detect peaks (using time-based distance) for cleaned vs not audio. using these parameter values
    # based on results with test data
    # - height ensures peaks have a minimum amplitude
    # - distance=int(samplingRate * 0.1) ensures peaks are at least 0.1 seconds apart
    # - prominence ensures peaks must stand out by at least "prominence" relative to their surroundings
    # clean_peaks, properties = detect_peaks_dynamic(
    #     audio_clean,
    #     sampling_rate,
    #     min_distance_sec= PEAK_DETECTION_PARAMS["distance_sec"],
    # )
    
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

    # Convert sample indices to time values for x-axis 
    time_clean = (np.arange(len(audio_clean)) + num_samples) / sampling_rate
    time = np.arange(len(audio)) / (sampling_rate)
    
    #plot and analyze intervals
    analyzeIntervals(clean_peaks, time_clean, pattern)

    plotPeaksComparison(time, time_clean, audio, audio_clean, sampling_rate, original_peaks, clean_peaks)

if __name__ == "__main__":
    main()