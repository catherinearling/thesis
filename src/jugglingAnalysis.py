from pathlib import Path
import os #for file path handling
import math
from analysisHelpers import *

from scipy.signal import find_peaks, medfilt
import numpy as np

import librosa
import soundfile as sf


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
    mask = medfilt(mask, kernel_size=(1,7))

    # apply mask to original data
    S_clean = S_full * mask

    # Inverse STFT to reconstruct cleaned audio by combining the filtered magnitude (S_clean) 
    # with the original phase data (phase)
    audio_clean = librosa.istft(S_clean * phase)

    return audio_clean


#------------------------------------------------------------
# Score a list of candidate cycle start times against detected catches
# using Gaussian PDF — returns avg log probability
#------------------------------------------------------------
def scoreCycles(predicted_cycle_starts, catch_times, sigma):
    log_prob_sum = 0.0
    for predicted_time in predicted_cycle_starts:
        nearest = nearestCatch(catch_times, predicted_time)
        diff = nearest - predicted_time
        p = pdf(diff, mu=0, sigma=sigma)
        log_prob_sum += math.log(p)
    return log_prob_sum / len(predicted_cycle_starts)


#------------------------------------------------------------
# Find the best cycle duration and predicted cycle start times
# by sliding a window of pattern_length across the catch times
#------------------------------------------------------------
def findBestCycles(catch_times, pattern_length):

    # Expected reasonable min and max interval .1 seconds, max interval .5 seconds
    min_interval = INTERVAL_BOUNDS["min"]
    max_interval = INTERVAL_BOUNDS["max"]

    # Identify catches outside expected bounds
    #irregular_intervals = (intervals < min_interval) | (intervals > max_interval)

    # ---------Estimate cycle length
    # every ball must be thrown and caught once per cycle
    # Rough estimate of cycle time based on initial catches
    # this is considering every single spike as a catch time!!!
    predicted_cycles = []
    best_avg_log_prob = 0 # Initialize to first guess inside loop
    initialized = False
    sigma = None
    best_cycle_duration = None

    # Use multiple guesses for cycle durations based on different starting catch pairs
    #consider each possible cycle length btwn ith catch and first catch -- start from i = pattern_length
    for i in range(pattern_length, len(catch_times)):
        begin = i - pattern_length 
        # Guess cycle duration with sliding window of cycle durations
        cycle_duration_guess = catch_times[i] - catch_times[begin]
        
        #if guess is within reasonable estimates, consider it
        if(cycle_duration_guess <= (max_interval * pattern_length)) \
            and (cycle_duration_guess >= (min_interval * pattern_length)):

            predicted_cycle_starts = []
            predicted_cycle_starts.append(catch_times[i])
            predicted_cycle_starts.append(catch_times[begin])
            
            curr_num_matches = 0
            #step forwards starting at end to find backwards matches for this cycle duration guess
            current_time = catch_times[i]
            while (current_time + cycle_duration_guess) <= catch_times[-1]:
                current_time += cycle_duration_guess
                predicted_cycle_starts.append(current_time)

                # # If we step one cycle length away from our current time, 
                # # do we land on a detected catch? (within tolerance)
                # closest = nearestCatch(catch_times, current_time)
                

            #step backwards starting at index i-pattern_length to find forwards matches for this cycle duration guess
            current_time = catch_times[begin]
            while (current_time - cycle_duration_guess) >= catch_times[0]:
                current_time -= cycle_duration_guess
                predicted_cycle_starts.insert(0, current_time)

                # # If we step one cycle length away from our current time, 
                # # do we land on a detected catch? (within tolerance)
                # closest = nearestCatch(catch_times, current_time)

            # --------- Scoring ---------
            # A candidate cycle is “good” if its predicted starts sit consistently 
            # close to real catch times
            
            # Choose sigma (the expected jitter around ideal times)
            sigma = TOLERANCE*0.85 #less strict for bad matches if its bigger
            avg_log_prob = scoreCycles(predicted_cycle_starts, catch_times, sigma)

            # Initialize best guess on first valid prediction
            #otherwise, keep this prediction if it has best accuracy so far, and is a valid accuracy (0 < acc < 1)
                #valid accuracy means we are not overpredicting the amount of cycles nor underpredicting
            if (avg_log_prob > best_avg_log_prob) or (not initialized):
                if not initialized:
                    initialized = True
                predicted_cycles = predicted_cycle_starts
                best_avg_log_prob = avg_log_prob
                best_cycle_duration = cycle_duration_guess

    return predicted_cycles, best_cycle_duration, sigma, best_avg_log_prob
    


#------------------------------------------------------------
# Try each realistic pattern length and return the one whose
# best cycle duration scores highest against the catch times.
# Returns (pattern_length, predicted_cycles, best_cycle_duration,
#          best_sigma, best_avg_log_prob)
#------------------------------------------------------------
def detectPatternLength(catch_times):
    best_result = None
    best_score = -np.inf

    for candidate_length in range(2, 6):  # siteswap lengths 2-5 are realistic
        if len(catch_times) < candidate_length:
            continue

        predicted_cycles, best_cycle_duration, best_sigma, avg_log_prob = \
            findBestCycles(catch_times, candidate_length)

        if best_sigma is None:
            continue

        print(f"  Pattern length {candidate_length}: score = {avg_log_prob:.4f}, "
              f"cycle duration = {best_cycle_duration:.3f}s")

        if avg_log_prob > best_score:
            best_score = avg_log_prob
            best_result = (candidate_length, predicted_cycles,
                           best_cycle_duration, best_sigma, avg_log_prob)

    return best_result



#------------------------------------------------------------
# Analyze the timing intervals between detected peaks to estimate rhythmic juggling cycles
# ----assumes the times are in order 
#  paramters:
# --peaks: indices of detected catch times / spikes
# --time: time values corresponding to audio samples
# --pattern: vanilla siteswap pattern (like 441, 51)
#returns list of predicted cycle start timestamps (in seconds) or an empty list if no
#valid cycle guesses found
#------------------------------------------------------------
def analyzeIntervals(peaks, time, pattern):
    #put intervals btwn catches into data struct
    #convert peak indices to times
    catch_times = time[peaks]

    if len(catch_times) == 0:
        print("No peaks detected. Try adjusting noise reduction or peak detection params.")
        return []
    print(f"Detected {len(catch_times)} catch times.")

    # ── Pattern known vs auto-detect ────────────────────────────────────────
    if pattern is not None:
        pattern_length = len(str(pattern))
        if len(catch_times) < pattern_length:
            print(f"Not enough catches detected for pattern length {pattern_length}.")
            return [], []

        predicted_cycles, best_cycle_duration, best_sigma, best_avg_log_prob = \
            findBestCycles(catch_times, pattern_length)

        if best_cycle_duration is None:
            print("Could not find any valid cycle guesses.")
            return [], []
    else:
        print("No pattern provided — attempting to detect pattern from audio...")
        pattern_length, predicted_cycles, best_cycle_duration, best_sigma, best_avg_log_prob = detectPatternLength(catch_times)

        if best_sigma is None:
            print("Could not determine pattern length from audio.")
            return [], []

        print(f"\nBest pattern length detected: {pattern_length}")
        
    #add predicted catches within each cycle
    # In a vanilla siteswap, throws are ideally evenly spaced one beat apart.
    # beat_duration = cycle_duration / pattern_length
    # so place one catch per beat position within the cycle.
    beat_duration = best_cycle_duration / pattern_length
    total_beats = len(predicted_cycles) * pattern_length
    predicted_catches = [predicted_cycles[0] + i * beat_duration for i in range(total_beats)]

    #now, predicted_catches should have all the predicted catch times based on the best cycle duration guess,
    # including cycle starts

    #only plot the best (in terms of accuracy) predictions we got
    # normalize accuracy so it's roughly 0 to 100%
    accuracy = math.exp(best_avg_log_prob) / pdf(0, 0, best_sigma) * 100
    print(f"Pattern match score (accuracy): {accuracy}%\n")


    # Coaching Tips ------------------------------------------------
    intervals = np.diff(catch_times)
    avg_interval = np.mean(intervals)
    std_interval = np.std(intervals)
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
    if estimated_cycles > 0 and (len(predicted_cycles) / estimated_cycles) < 0.6:
        print("The detected pattern only lines up with part of the run. "
                "This could mean extra out-of-rhythm throws or drops between clean cycles.")

    plotCycles(catch_times, predicted_cycles, predicted_catches)

    return predicted_cycles


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
    
    ############ write out detected peaks to txt file #####################

    # Convert sample indices to time values for x-axis
    time_clean = (np.arange(len(audio_clean)) + num_samples) / sampling_rate

    # ---- Export detected catch/peak timestamps ----
    # timestamps in seconds for each detected peak in cleaned audio
    peak_times_sec = time_clean[clean_peaks]

    # output file: same base name as audio, but .txt, saved next to the audio (data/)
    audio_stem = Path(filename).stem
    out_path = Path("data") / f"{audio_stem}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for t in peak_times_sec:
            f.write(f"{t:.6f},\n")  # 6 decimals = microsecond-ish resolution

    print(f"Wrote {len(peak_times_sec)} peak timestamps to {out_path}")
    #################

    #plot and analyze intervals
    predicted_cycles = analyzeIntervals(clean_peaks, time_clean, pattern)

    if predicted_cycles:
        pattern_out_path = Path("data") / f"{audio_stem}-pattern.txt"
        with open(pattern_out_path, "w", encoding="utf-8") as f:
            for t in sorted(predicted_cycles):
                f.write(f"{t:.6f},\n")
        print(f"Wrote {len(predicted_cycles)} predicted cycle timestamps to {pattern_out_path}")
    else:
        print("No predicted cycle timestamps to write.")
    
    time = np.arange(len(audio)) / (sampling_rate)
    plotPeaksComparison(time, time_clean, audio, audio_clean, sampling_rate, original_peaks, clean_peaks)

if __name__ == "__main__":
    main()