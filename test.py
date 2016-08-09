from __future__ import print_function
import vamp
import librosa
import numpy as np
import matplotlib.pyplot as plt 
import sys
import scipy.signal as sps

audio_file_name = sys.argv[1]
bpm = sys.argv[2]
audio, sr = librosa.load(audio_file_name, sr=44100, mono=True)

data = vamp.collect(audio, sr, "mtg-melodia:melodia")

hop, melody = data['vector']

# 44100/128 timestamps per second

timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)

# plot extracted frequency without zero
melody_pos = melody[:]
melody_pos[melody<=0] = 0
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_pos)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.savefig(audio_file_name.rsplit(".", 1)[0] + 'freqs.jpeg')

# convert frequencies to notes, relative to A4 = 440 Hz
melody_notes = 12 * np.log2(melody_pos/440)

# convert notes to MIDI numbers, where A4 = 69
melody_notes = melody_notes + 69

# get rid of negative infinity notes
melody_notes[melody_notes < 0] = 0

# round note MIDI numbers to nearest integer, to account for vibrato
melody_notes = np.round(melody_notes)

# plot midi notes
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_notes)
plt.xlabel('Time (s)')
plt.ylabel('MIDI Note Number')
plt.savefig(audio_file_name.rsplit(".", 1)[0] + 'rough.jpeg')

# smooth out using scipy's median filter algorithm
# we want a window of size = 0.5 seconds, so we convert that to timestamps
filter = int(0.25 * 44100.0/128.0)

# make sure filter size is odd (medfilt requirement)
if filter % 2 == 0:
    filter = filter + 1

melody_notes_smooth = sps.medfilt(melody_notes, filter)

# plot smooth midi notes
plt.figure(figsize=(18,6))
plt.plot(timestamps, melody_notes_smooth)
plt.xlabel('Time (s)')
plt.ylabel('MIDI Note Number')
plt.savefig(audio_file_name.rsplit(".", 1)[0] + 'smooth.jpeg')

# extract individual discrete midi numbers from smooth line
# while keeping track of duration of each note
indiv_notes = []
numStamps = 0
for n in range(1, len(melody_notes_smooth)):
    if melody_notes_smooth[n] != melody_notes_smooth[n-1]:
        indiv_notes.append((melody_notes_smooth[n-1], (numStamps+1) * 128.0/44100.0))
        numStamps = 0
    else:
        numStamps = numStamps + 1

# narrow down valid discrete midi numbers
# by removing a note if its duration is too small
for n in range(1, len(indiv_notes)):
    if indiv_notes[n][1] < 0.1:
        indiv_notes[n] = list(indiv_notes[n])
        #indiv_notes[n-1] = list(indiv_notes[n-1])
        #indiv_notes[n-1][1] += indiv_notes[n][1]
        indiv_notes[n][1] = 0

# convert discrete midi numbers to actual notes in letter form
actual_notes = indiv_notes[:]
note_letters = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

for n in range(len(indiv_notes)):
    if indiv_notes[n][1] != 0:
        actual_notes[n] = (note_letters[int(indiv_notes[n][0] % 12)], indiv_notes[n][1])
    else:
        actual_notes[n] = (0, 0)

for note in actual_notes:
    if note[1] != 0:
        print(note)
        
