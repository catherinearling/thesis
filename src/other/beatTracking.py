# Beat tracking example
import librosa

# 1. Get the file path to an included audio example
filename = librosa.example('nutcracker')


# load and decode the audio as a time series y, 
# represented as a one-dimensional NumPy floating point array. 
# The variable sr contains the sampling rate of y----the number of samples per second of audio.
y, sr = librosa.load(filename)

# 3. Run the default beat tracker
#The output of the beat tracker is an estimate of the tempo (in beats per minute), 
# and an array of frame numbers corresponding to detected beat events.
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

#print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
print('Estimated tempo: {:.2f} beats per minute'.format(tempo[0]))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)