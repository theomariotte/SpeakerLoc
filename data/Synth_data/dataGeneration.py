# VIBERT Samuel & OUAKAN Mohamed
# 5A VA-alt
# Projet 5A : AMÉLIORATION DE LA TRANSCRIPTION DE RÉUNION PAR LOCALISATION DE LOCUTEUR
# Simulation avec des échantillons de la réunion. Ici, les 4 fichiers audios (headset) représentent quelques secondes de chaque locuteur
# Le 5ème fichier représente le bruit de fond

########################################################################################################################

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
#Dimensions de la salle + TR60 :
rt60_tgt = 0.3  # en secondes
room_dim = [4, 6]  # en mètres
input_dir = "data/Synth_data/input/"
output_dir = "data/Synth_data/output/"

# choose a target sampling rate
fs = 16000
# number of sources
nb_src = 4

# microphone array properties
nb_mic = 8
radius = 100e-3
Lg_t = 0.100                # Largeur du filtre (en s)
Lg = np.ceil(Lg_t*fs)       # en échantillons
center = [2, 3]
fft_len = 512

# signal to noise ratio
snr = 10.
ref_mic_idx = 0
# if True, add noise source in a corner of the room in addition to noise added by simulation
noise_src_fl = True

# overlap between signals
overlap12 = 1.0
overlap23 = 5.0
overlap34 = 3.0

# check directories
if not os.path.exists(input_dir):
    os.mkdir(input_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Importation des 4 signaux audios :
fs_audio, audio1 = wavfile.read(f"{input_dir}is1000a_Headset 0_mono.wav")
_, audio2 = wavfile.read(f"{input_dir}is1000a_Headset 1_mono.wav")
_, audio3 = wavfile.read(f"{input_dir}is1000a_Headset 2_mono.wav")
_, audio4 = wavfile.read(f"{input_dir}is1000a_Headset 3_mono.wav")


# adjust sampling rate if audio does not fit targeted one
if fs_audio > fs:
    audio1 = dsp.decimate(x=audio1,q=fs_audio//fs)
    audio2 = dsp.decimate(x=audio2,q=fs_audio//fs)
    audio3 = dsp.decimate(x=audio3,q=fs_audio//fs)
    audio4 = dsp.decimate(x=audio4,q=fs_audio//fs)


# Théorème de Sabine pour les paramètres de la méthode source-image :
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Création de la salle :
room = pra.ShoeBox(room_dim,
                   fs=fs,
                   materials=pra.Material(e_absorption),
                   max_order=max_order)

# Placement des sources dans la salle (les sources sont juxtaposées à l'aide du paramètre Delay) :
# adjust delay with respect to signal duration and overlap between sources
start = 0.5
room.add_source([1, 2], signal=audio1, delay=start)
start += len(audio1)/fs - overlap12
room.add_source([1, 4], signal=audio2, delay=start)
start+= len(audio2)/fs - overlap23
room.add_source([3, 2], signal=audio3, delay=start)
start+=len(audio3)/fs - overlap34
room.add_source([3, 4], signal=audio4, delay=start)

# for file naming and noise generation (if needed)
total_duration = start+len(audio4)/fs
ola_ratio = int(np.ceil((overlap12+overlap23+overlap34)/total_duration))

# in case of noise source added
if noise_src_fl:
    # Ajout d'un bruit de fond, d'une durée égale à la somme des 4 audios précédents
    fs_noise, bruit = wavfile.read(f"{input_dir}bruit.wav")
    if fs_noise > fs:
        bruit = dsp.decimate(x=bruit,
                            q=fs_noise//fs)

    noise_duration = int(np.ceil(total_duration*fs))
    room.add_source([3.9, 5.9], signal=bruit[:noise_duration], delay=0.5)
    nb_src+=1


# create microphone array
micropnts = np.zeros((8, 2))
micropnts[0, :] = np.array([-0.1, 0])
micropnts[1, :] = np.array([-0.1*np.sqrt(2)/2, -0.1*np.sqrt(2)/2])
micropnts[2, :] = np.array([0, -0.1])
micropnts[3, :] = np.array([0.1*np.sqrt(2)/2, -0.1*np.sqrt(2)/2])
micropnts[4, :] = np.array([0.1, 0])
micropnts[5, :] = np.array([0.1*np.sqrt(2)/2, 0.1*np.sqrt(2)/2])
micropnts[6, :] = np.array([0, 0.1])
micropnts[7, :] = np.array([-0.1*np.sqrt(2)/2, 0.1*np.sqrt(2)/2])
micropnts += center

mics = pra.beamforming.Beamformer(np.array(micropnts).T, fs, N=fft_len, Lg=Lg, hop=None)

"""
#TODO (21/03/08) - seek why M=nb_mic-1 to get 8 microphones 
echo = pra.circular_2D_array(center=center, M=nb_mic-1, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room.fs, N=fft_len, Lg=Lg)
"""

# Placement de l'antenne :

room.add_microphone_array(mics)

# Affichage de la salle :

fig, ax = room.plot()

# Graphiques Beamforming :

room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-nb_src][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 1")
room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-nb_src+1][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 2")
room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-nb_src+2][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 3")
room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-nb_src+3][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 4")
if noise_src_fl:
    room.mic_array.rake_delay_and_sum_weights(room.sources[nb_src-nb_src+4][:1])
    fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
    ax.legend(['500', '1000', '2000', '4000', '8000'])
    plt.title("Source de bruit")

plt.show()
# Simulation (Construit la RIR automatiquement) :

room.simulate(reference_mic=ref_mic_idx,snr=snr)

# Enregistrement du ignal audio reçu par l'antenne :
name = "IS1000a_TR{:d}_T{:d}_nch{:d}_ola{:d}_noise{:d}".format(int(rt60_tgt*1e3),int(np.ceil(total_duration)),int(nb_mic),ola_ratio,int(noise_src_fl))
room.mic_array.to_wav(
    f"{output_dir}{name}.wav",
    norm=True,
    bitdepth=np.int16)

# Amélioration du résultat grâce au beamforming :
signal_das = mics.process(FD=False)
# Enregistrement du ignal audio reçu par l'antenne :
name = "BF_IS1000a_TR{:d}_T{:d}_nch{:d}_ola{:d}_noise{:d}".format(int(rt60_tgt*1e3),int(np.ceil(total_duration)),int(nb_mic),ola_ratio,int(noise_src_fl))
wavfile.write(filename=f"{output_dir}{name}.wav",
              rate=fs,
              data=signal_das)

print("DAS Beamformed Signal:")
IPython.display.Audio(signal_das, rate=fs)

# Création du fichier texte regroupant les données de l'étude du beamforming (en cours...) :

# Permet d'afficher la matrice complète
np.set_printoptions(threshold=sys.maxsize)

# Caractéristiques du signal et récupération :

# Affiche le nombre de canal du signal ainsi que le nombre d'échantillons
print(room.mic_array.signals.shape)
# Associe à M le nombre de canaux du signal, et à N le nombre d'échantillons du signal
M, N = room.mic_array.signals.shape

# Graphiques :

plt.figure()

# On s'assure qu'il y a bien une différence entre les signaux de chaque microphone :

# Ici on regarde la différence entre le microphone 1 de l'antenne et le cinquième
tmp = np.sum(room.mic_array.signals[1, :] -
             room.mic_array.signals[5, :], axis=0)
tmp /= N
print(f"Ecart moyen = {tmp}")

# Mesure du TR60 :

rt60 = room.measure_rt60()
print("Le TR60 désiré est {}".format(rt60_tgt))  # TR60 choisi
print("Le TR60 mesuré est {}".format(rt60[1, 0]))  # TR60 mesuré

# Cohérence des signaux :
"""
plt.subplot(4, 2, 1)
cor = plt.cohere(room.mic_array.signals[1, :], room.mic_array.signals[0, :])
plt.title("Microphone 1")
plt.xlabel(" ")
plt.subplot(4, 2, 2)
cor = plt.cohere(room.mic_array.signals[2, :], room.mic_array.signals[0, :])
plt.title("Microphone 2")
plt.xlabel(" ")
plt.subplot(4, 2, 3)
cor = plt.cohere(room.mic_array.signals[3, :], room.mic_array.signals[0, :])
plt.title("Microphone 3")
plt.xlabel(" ")
plt.subplot(4, 2, 4)
cor = plt.cohere(room.mic_array.signals[4, :], room.mic_array.signals[0, :])
plt.title("Microphone 4")
plt.xlabel(" ")
plt.subplot(4, 2, 5)
cor = plt.cohere(room.mic_array.signals[5, :], room.mic_array.signals[0, :])
plt.title("Microphone 5")
plt.xlabel(" ")
plt.subplot(4, 2, 6)
cor = plt.cohere(room.mic_array.signals[6, :], room.mic_array.signals[0, :])
plt.title("Microphone 6")
plt.xlabel(" ")
plt.subplot(4, 2, 7)
cor = plt.cohere(room.mic_array.signals[7, :], room.mic_array.signals[0, :])
plt.title("Microphone 7")
plt.xlabel(" ")
plt.subplot(4, 2, 8)
cor = plt.cohere(room.mic_array.signals[8, :], room.mic_array.signals[0, :])
plt.title("Microphone 8")
plt.xlabel(" ")
plt.show()

# Signal reçu par chaque microphone :

# On crée un vecteur temps pour pour tracer les signaux en fonction du temps et non en fonction du nombre d'échantillons
T = np.linspace(0, (N-1)/fs, N)
plt.subplot(4, 2, 1)
plt.plot(T, room.mic_array.signals[1, :])
plt.title("Microphone 1")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 2)
plt.plot(T, room.mic_array.signals[2, :])
plt.title("Microphone 2")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 3)
plt.plot(T, room.mic_array.signals[3, :])
plt.title("Microphone 3")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 4)
plt.plot(T, room.mic_array.signals[4, :])
plt.title("Microphone 4")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 5)
plt.plot(T, room.mic_array.signals[5, :])
plt.title("Microphone 5")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 6)
plt.plot(T, room.mic_array.signals[6, :])
plt.title("Microphone 6")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 7)
plt.plot(T, room.mic_array.signals[7, :])
plt.title("Microphone 7")
plt.xlabel("Time [s]")
plt.subplot(4, 2, 8)
plt.plot(T, room.mic_array.signals[8, :])
plt.title("Microphone 8")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()
"""
