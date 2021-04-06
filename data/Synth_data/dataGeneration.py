# VIBERT Samuel & OUAKAN Mohamed
# 5A VA-alt
# Projet 5A : AMÉLIORATION DE LA TRANSCRIPTION DE RÉUNION PAR LOCALISATION DE LOCUTEUR
# Simulation avec des échantillons de la réunion. Ici, les 4 fichiers audios (headset) représentent quelques secondes de chaque locuteur
# Le 5ème fichier représente le bruit de fond

########################################################################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import IPython
import sys
import os
import wave
import scipy.signal as dsp

"""
Parameters
"""
### Repositories
# Where source signals are stored
input_dir = "data/Synth_data/input/"
# Where simulation results are stored
output_dir = "data/Synth_data/output/CIRC/"

### Simulation parameters
# Target sampling rate [Hz] (resampling if needed)
fs = 16000
# RT60 [s]
rt60_tgt = 0.3  # en secondes
room_dim = [4, 6]  # en mètres
# Signal to noise ratio
snr = None
ref_mic_idx = 0
# if True, add noise source in a corner of the room in addition to noise added by simulation
noise_src_fl = False
noise_src_loc = [3.9, 5.9]

### Sources DOAs [deg]
doa_src = [45., 135. ,225. ,315.]
#doa_src = [45.0]
# overlap between the sources above
#overlap_list = [5.0]
overlap_list = [1.0, 4.0, 2.0]
# distance from array center [m]
src_dist = 1.

### microphone array properties

# type of array ("circular","ULA")
nb_mic = 8

# if circular :
theta_start = 0
theta_stop = 360
radius = 0.1

Lg_t = 0.100                # Largeur du filtre (en s)
Lg = np.ceil(Lg_t*fs)       # en échantillons
center = [2, 3]
fft_len = 512

# overlap between signals
overlaps = np.zeros((len(overlap_list)+1,))
overlaps[:-1] += np.array(overlap_list)

"""
Simulation
"""
# check directories
if not os.path.exists(input_dir):
    os.mkdir(input_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Importation des signaux audios (up to 4):
nb_src = len(doa_src)
doa_src = np.array(doa_src, dtype=np.float)
i = 0
audio = []
while i < nb_src:
    fs_audio, sig = wavfile.read(f"{input_dir}is1000a_Headset {i}_mono.wav")

    # adjust sampling rate if audio does not fit targeted one
    if fs_audio > fs:
        sig = dsp.decimate(x=sig, q=fs_audio//fs)

    audio.append(sig)
    i += 1


# Théorème de Sabine pour les paramètres de la méthode source-image :
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Création de la salle :
room = pra.ShoeBox(room_dim,
                   fs=fs,
                   materials=pra.Material(e_absorption),
                   max_order=max_order)


# Source positions
doa_src *= np.pi/180.
srcpnts = np.zeros((nb_src, 2))
start = 0.5
for i, doa in enumerate(doa_src):

    # cartesian coordinates
    x_ = src_dist * np.cos(doa)
    y_ = src_dist * np.sin(doa)

    # source location in the array frame of reference
    srcpnt = center+np.array([x_, y_])

    # add source to room object
    room.add_source(srcpnt, signal=audio[i], delay=start)

    # update start time (including overlaps)
    start += len(audio[i])/fs - overlaps[i]

# for file naming and noise generation (if needed)
total_duration = start
ola_ratio = int(np.ceil(np.sum(overlaps)/total_duration))

# in case of noise source added
if noise_src_fl:
    # Ajout d'un bruit de fond, d'une durée égale à la somme des 4 audios précédents
    fs_noise, bruit = wavfile.read(f"{input_dir}bruit.wav")
    if fs_noise > fs:
        bruit = dsp.decimate(x=bruit,
                             q=fs_noise//fs)

    noise_duration = int(np.ceil(total_duration*fs))
    room.add_source(noise_src_loc, signal=bruit[:noise_duration], delay=0.5)
    nb_src += 1

# create circular microphone array
theta_step = (theta_stop - theta_start)/nb_mic
theta = np.arange(theta_start, theta_stop, theta_step)
theta *= np.pi/180.0
x_mic = radius * np.cos(theta)
y_mic = radius * np.sin(theta)
micropnts = np.array([x_mic, y_mic]).T
micropnts += center

mics = pra.beamforming.Beamformer(
    np.array(micropnts).T, fs, N=fft_len, Lg=Lg, hop=None)

# Placement de l'antenne :
room.add_microphone_array(mics)
# show room
room.plot()

# Graphiques Beamforming :
for ii in range(nb_src):
    room.mic_array.rake_delay_and_sum_weights(room.sources[ii][:1])
    fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
    ax.legend(['500', '1000', '2000', '4000', '8000'])
    plt.title(f"Source {ii+1}")

if noise_src_fl:
    room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-1][:1])
    fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
    ax.legend(['500', '1000', '2000', '4000', '8000'])
    plt.title("Source de bruit")

plt.show()

# Simulation (Construit la RIR automatiquement) :
if snr is not None:
    room.simulate(reference_mic=ref_mic_idx, snr=snr)
else:
    room.simulate()
    snr = "inf"

"""
Saving and post processing
"""
# Enregistrement du signal audio reçu par l'antenne :
name = "IS1000a_T{:d}_nch{:d}_snr{}_ola{:d}_noise{:d}".format(int(np.ceil(
                                                                         total_duration)),
                                                                     int(nb_mic),
                                                                     str(snr),
                                                                     ola_ratio,
                                                                     int(noise_src_fl))

# speakers segments
segs=list()
t_start=0.5
for ii in range (len(doa_src)):
    segs.append({"speaker": f"spk{ii+1}", "start":t_start , "stop":t_start+len(audio[ii])/fs ,"doa": doa_src[ii]*180.0/np.pi})
    t_start += len(audio[ii])/fs - overlaps[ii]

with open(output_dir+name+'.pkl','wb') as fh:
    pickle.dump(segs,fh)

room.mic_array.to_wav(
    f"{output_dir}{name}.wav",
    norm=True,
    bitdepth=np.int16)

# Amélioration du résultat grâce au beamforming :
signal_das = mics.process(FD=False)
signal_das /= np.max(np.abs(signal_das))
# Enregistrement du ignal audio reçu par l'antenne :
name = "BF_"+name
wavfile.write(filename=f"{output_dir}{name}.wav",
              rate=fs,
              data=signal_das)

#TODO Création du fichier texte regroupant les données de l'étude du beamforming (en cours...) :

# Affiche le nombre de canal du signal ainsi que le nombre d'échantillons
print(f"Signal shape : {room.mic_array.signals.shape}")

# Mesure du TR60 :
rt60 = room.measure_rt60()
print("Le TR60 désiré est {}".format(rt60_tgt))  # TR60 choisi
print("Le TR60 mesuré est {}".format(rt60[1, 0]))  # TR60 mesuré
