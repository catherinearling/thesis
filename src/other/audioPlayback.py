# Required libraries
import numpy as np
import librosa

#libraries supported by VS Code
import sounddevice as sd 
import soundfile as sf

# Sampling rate
sr = 22050

# Generate a sine sweep from C3 to C5
y_sweep = librosa.chirp(fmin=librosa.note_to_hz('C3'),
                        fmax=librosa.note_to_hz('C5'),
                        sr=sr,
                        duration=1)

# Play audio using sounddevice
print("Playing generated sweep sound...")
sd.play(y_sweep, samplerate=sr)
sd.wait()  # Wait for playback to finish

# Load an example audio file
y, sr = librosa.load(librosa.ex('trumpet'))

# Play the loaded audio file
print("Playing trumpet sound...")
sd.play(y, samplerate=sr)
sd.wait()

# Save the trumpet sound as a WAV file
sf.write('trumpet_output.wav', y, sr)
print("Saved 'trumpet_output.wav'. You can play it manually if needed.")

# Sonify pitch estimates
f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'),
                                             fill_na=None)

times = librosa.times_like(f0)
vneg = (-1)**(~voiced_flag)  # Handle unvoiced regions
y_f0 = librosa.tone(f0 * vneg, sr=sr)  # Alternative to mir_eval.sonify

# Play the sonified pitch contour
print("Playing sonified pitch contour...")
sd.play(y_f0, samplerate=sr)
sd.wait()

# Detect onsets and create click track
onset_env = librosa.onset.onset_strength(y=y, sr=sr, max_size=5)
onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
y_clicks = librosa.clicks(times=onset_times, length=len(y), sr=sr)

# Play original sound with clicks
print("Playing trumpet with detected onsets (clicks)...")
sd.play(y + y_clicks, samplerate=sr)
sd.wait()

# Save the output with clicks
sf.write('trumpet_with_clicks.wav', y + y_clicks, sr)
print("Saved 'trumpet_with_clicks.wav'. You can play it manually if needed.")
