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

    # noise_mean = np.mean(noise_region, axis=1)
    # noise_std  = np.std(noise_region, axis=1)
    # noise_margin_std = 1.2  # how much above the mean noise power to set threshold
    # threshold is mean + k * std, so only clearly-above-noise stuff survives
    #noise_threshold = noise_mean + noise_margin_std * noise_std


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
        nearest, _idx = nearestCatch(catch_times, predicted_time)
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
            
            #step forwards starting at end to find backwards matches for this cycle duration guess
            current_time = catch_times[i]
            while (current_time + cycle_duration_guess) <= catch_times[-1]:
                current_time += cycle_duration_guess
                predicted_cycle_starts.append(current_time)

            #step backwards starting at index i-pattern_length to find forwards matches for this cycle duration guess
            current_time = catch_times[begin]
            while (current_time - cycle_duration_guess) >= catch_times[0]:
                current_time -= cycle_duration_guess
                predicted_cycle_starts.insert(0, current_time)

            # --------- Scoring ---------
            # A candidate cycle is “good” if its predicted starts sit consistently 
            # close to real catch times
            
            # Choose sigma (the expected jitter around ideal times)
            sigma = TOLERANCE*0.85 #less strict for bad matches if its bigger
            avg_log_prob = scoreCycles(predicted_cycle_starts, catch_times, sigma)

            # Initialize best guess on first valid prediction
            #otherwise, keep this prediction if it has best accuracy so far
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

    for candidate_length in range(1, 6):  # siteswap lengths 2-5 are realistic
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
    if pattern is None:
        print("No pattern provided — attempting to detect pattern from audio...")
        pattern_length, predicted_cycles, best_cycle_duration, best_sigma, best_avg_log_prob = detectPatternLength(catch_times)

        if best_sigma is None:
            print("Could not determine pattern length from audio.")
            return [], []

        print(f"\nBest pattern length detected: {pattern_length}")

    else:
        pattern_length = len(str(pattern))
    
    if len(catch_times) < pattern_length:
        print(f"Not enough catches detected for pattern length {pattern_length}.")
        return [], []
    predicted_cycles, best_cycle_duration, best_sigma, best_avg_log_prob = \
        findBestCycles(catch_times, pattern_length)

    if best_cycle_duration is None:
        print("Could not find any valid cycle guesses.")
        return [], []
        
    #add predicted catches within each cycle
    # In a vanilla siteswap, throws are ideally evenly spaced one beat apart.
    # beat_duration = cycle_duration / pattern_length
    # so place one catch per beat position within the cycle.
    beat_duration = best_cycle_duration / pattern_length
    total_beats = len(predicted_cycles) * pattern_length
    predicted_catches = [predicted_cycles[0] + i * beat_duration for i in range(total_beats)]

    #now, predicted_catches should have all the predicted catch times based on the best cycle duration guess,
    # including cycle starts

    # Coaching Tips ------------------------------------------------
    print("Coaching Tips:")

    # Compute real timing error: residuals between each actual catch and its nearest predicted catch
    predicted_arr = np.array(predicted_catches)

    # These two lists will be built in parallel — one entry per detected catch.
    # remainders: how far off (in seconds) each actual catch was from its ideal predicted time
    # beat_positions: which beat slot within the pattern cycle that catch belongs to (0, 1, 2, 0, 1, 2, ...)
    remainders = []
    beat_positions = []
    
    for ct in catch_times:
        nearest_predicted, nearest_idx = nearestCatch(predicted_arr, ct)

        # Positive means the juggler caught it LATE (after the ideal moment).
        # Negative means they caught it EARLY (before the ideal moment).
        remainders.append(ct - nearest_predicted)
                
        # nearest_idx % pattern_length gives which beat slot within the cycle this catch
        # aligns to. We don't know which siteswap throw value that corresponds to because
        # the cycle anchor could start at any throw in the pattern — so we just label by
        # position number
        beat_positions.append(nearest_idx % pattern_length)
        
    remainders = np.array(remainders)
    beat_positions = np.array(beat_positions)

    # 1) analyze each beat position in the cycle separately, since patterns
    # can be made of throws of differing heights/durations. we care
    # about whether the juggler is consistent about their variation 
    #   like in a 441, consistent "long long short" is okay 
    #   but having it be "long long short short long short long long long" would be wrong
    flagged_any = False
    for pos in range(pattern_length):

        # Boolean mask: True only for catches that aligned to this beat position.
        #      if beat_positions = [0, 1, 2, 0, 1, 2] and pos=1,
        #      mask =              [F, T, F, F, T, F] 
        mask = beat_positions == pos
        
        # Pull out only the residuals for this beat position using the mask.
        pos_residuals = remainders[mask]

        # Skip if we don't have enough data points to say anything meaningful
        #    fewer than 3 samples, std and mean are too noisy to act on.
        if len(pos_residuals) < 3:
            continue


        # std (standard deviation) measures spread: how much the timing varies
        # from catch to catch at this position. 
        # High spread means sometimes early, sometimes late.
        #       the throw height is inconsistent.
        # Multiply by 1000 to convert seconds -> milliseconds for readability.
        spread_ms = np.std(pos_residuals) * 1000   # real ms spread for this beat position

        # mean measures systematic bias — if the average residual is consistently
        # positive, the juggler always catches late here; negative means always early.
        # Bias and spread are different problems: you can be consistently late with
        # tight spread (same mistake every time) or randomly scattered with low bias
        bias_ms = np.mean(pos_residuals) * 1000  # systematic early/late bias


        # Flag inconsistency from catch to catch
        if spread_ms > 100:
            flagged_any = True
            print(f"- Beat position {pos+1} (of {pattern_length}) is inconsistent "
                  f"(±{spread_ms:.0f}ms). Focus on making this throw the same height each time.")

        # Flag systematic bias 
        # The cause of a catch landing late or early is almost always the PRECEDING
        # throw — if you threw too high, gravity takes longer and the catch comes late.
        if abs(bias_ms) > 40:
            direction = "late" if bias_ms > 0 else "early"
            flagged_any = True
            print(f"- Beat position {pos+1} (of {pattern_length}) consistently lands "
                  f"{direction} by ~{abs(bias_ms):.0f}ms. "
                  "This usually means the preceding throw is mis-timed.")

    # If neither check triggered for any beat position, the juggler is doing well.
    if not flagged_any:
        print("- Great consistency across all throw positions!")
    
    # 2) Tempo drift
    intervals = np.diff(catch_times)
    if len(intervals) >= 6:
        third = len(intervals) // 3
        early_mean = np.mean(intervals[:third])
        mid_mean   = np.mean(intervals[third:2*third])
        late_mean  = np.mean(intervals[2*third:])
        drift = (late_mean - early_mean) / mid_mean
        if drift > 0.15:
            print(f"- You're slowing down as you go ({drift*100:.0f}% tempo drop). "
                  "Try to lock into a steady pace from the very first throw.")
        elif drift < -0.15:
            print(f"- You're speeding up over time ({abs(drift)*100:.0f}% tempo increase). "
                  "Rushing is usually a sign of tension — relax your throws.")

    # 3) Phase slip detection
    # A phase slip looks like: good alignment, then a stretch of misses, then good alignment again.
    # This is different from general inconsistency (which is spread randomly throughout).
    # We detect it by checking whether the misses are clustered together in time,
    # rather than scattered evenly across the run.

    # Classify each catch as a hit or miss based on TOLERANCE
    hit_mask = np.abs(remainders) <= TOLERANCE  # True = on time, False = off beat

    # Only worth checking if there's a meaningful number of misses to cluster
    miss_rate = 1 - np.mean(hit_mask)
    if miss_rate > 0.3:  # more than 30% of catches are off-beat
        
        # Split the run into thirds and check where the misses are concentrated
        n = len(hit_mask)
        third = n // 3
        miss_rate_early = 1 - np.mean(hit_mask[:third])
        miss_rate_mid   = 1 - np.mean(hit_mask[third:2*third])
        miss_rate_late  = 1 - np.mean(hit_mask[2*third:])

        # If one third is much worse than the others, misses are clustered — phase slip
        miss_rates = [miss_rate_early, miss_rate_mid, miss_rate_late]
        worst = max(miss_rates)
        best  = min(miss_rates)

        if worst > best + 0.4:  # large contrast between sections
            worst_section = ["early", "middle", "late"][miss_rates.index(worst)]
            print(f"- There's a cluster of off-beat catches in the {worst_section} portion of the run. "
                  "This could be a single hiccup (a dropped ball, an extra throw, or one "
                  "badly timed catch) that threw off the cycle alignment for a stretch — "
                  "it doesn't necessarily mean your rhythm was bad the whole time.")


    # 4) Overall summary
    # Accuracy: what fraction of catches landed within TOLERANCE of their predicted beat.
    hits = np.sum(np.abs(remainders) <= TOLERANCE)
    accuracy = (hits / len(catch_times)) * 100
    print(f"\nOverall pattern match score: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("Your timing matches the pattern very well — keep it up!")
    elif accuracy >= 50:
        print("You're roughly following the pattern. Focus on the tips above to sharpen your timing.")
    else:
        print("The timing doesn't strongly match a repeating pattern yet. "
              "Try a slower pace and aim for even, relaxed throws.")
        
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