from scipy.io.wavfile import read
import numpy as np

def calc_distances(sound_file):
    min_val = 5000
    fs, data = read(sound_file)

    # Handle stereo audio
    if len(data.shape) > 1:
        data = data[:, 0]  # Use only the first channel

    data_size = len(data)
    focus_size = int(0.15 * fs)
    focuses = []
    distances = []
    idx = 0

    while idx < data_size:
        if data[idx] > min_val:
            mean_idx = idx + focus_size // 2
            focuses.append(mean_idx / fs)  # Now it's in seconds
            if len(focuses) > 1:
                distances.append(focuses[-1] - focuses[-2])
            idx += focus_size  # Skip ahead to avoid double-detecting the same knock
        else:
            idx += 1
    return distances

def accept_test(pattern, test, min_error):
    if len(pattern) > len(test):
        return False
    for i, dt in enumerate(pattern):
        if abs(dt - test[i]) >= min_error:
            return False
    return True

pattern = calc_distances('knock.wav')
test = calc_distances('knock2.wav')
min_error = 0.1  # in seconds

print(accept_test(pattern, test, min_error))
print(pattern)
