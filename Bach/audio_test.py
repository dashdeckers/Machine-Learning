import time
import numpy as np
import simpleaudio as sa
import itertools as it

def find_ranges(lst, n=2):
    """ Return ranges for `n` or more repeated values.

    Taken from:
	https://stackoverflow.com/a/44792205

	Example:
	arr = np.array([1, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 4, 4, 2, 2])
	list(find_ranges(arr, 1))

	[(0, 1), (2, 5), (6, 6), (7, 7), (8, 11), (12, 13), (14, 15)]

    """
    groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if len(idx_g) >=n)
    return ((sub[0][0], sub[-1][0]) for sub in repeated)

def pianokeys_to_soundvectors(voices):
	base_freq = 440
	# Sample rates lower than 44100 are not accepted by simpleaudio
	sample_rate = 44100
	dur_per_sym = 1/20
	ticks_per_sym = int(np.floor(sample_rate * dur_per_sym))

	sound_vectors = list()

	# Iterate over the column vectors
	for voice in voices.T:
		# Initialize the sound vector
		sound_vec = np.zeros(len(voice) * ticks_per_sym)
		# Find the smallest non-zero symbol
		min_symbol = min([key for key in np.unique(voice) if key > 0])

		# Iterate over ranges of identical symbols
		for tone_range in find_ranges(voice, 1):
			# Start and end indices of identical symbols
			start_idx, stop_idx = tone_range
			# The symbol
			symbol = voice[start_idx]
			# Scale the indices with time
			start_idx *= ticks_per_sym
			stop_idx *= ticks_per_sym

			# Calculate the sine wave notes
			frequency = base_freq * 2**((symbol - min_symbol) / 12)
			for t in range(start_idx, stop_idx + 1):
				sound_vec[t] = np.sin(2 * np.pi * frequency * t / sample_rate)

		# Normalize to 16bit (required by simpleaudio)
		sound_vec = sound_vec * (2**15 - 1) / np.max(np.abs(sound_vec))
		sound_vec = sound_vec.astype(np.int16)

		sound_vectors.append(sound_vec)

	return sound_vectors

if __name__ == '__main__':
	t0 = time.time()

	with open('F.txt', 'r') as voice_file:
		voices = np.loadtxt(voice_file)

	print(f'Loaded data ({time.time() - t0})')

	sound_vectors = pianokeys_to_soundvectors(voices)

	print(f'Processed data ({time.time() - t0})')

	play_obj = sa.play_buffer(sound_vectors[3], 1, 2, 44100)
	play_obj.wait_done()
	# time.sleep(5)
	# play_obj.stop()