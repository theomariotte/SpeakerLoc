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
import wave

#Dimensions de la salle + TR60 :

rt60_tgt = 0.3  # en secondes
room_dim = [4, 6]  # en mètres
input_dir = "./input/"
output_dir = "./output/"

# Importation des 4 signaux audios :
# Ici, fs=48000 Hz. Pour les fichiers audios issus du corpus AMI, fs=16000 Hz

fs, audio1 = wavfile.read(f"{input_dir}is1000a_Headset 0_mono.wav")
fs, audio2 = wavfile.read(f"{input_dir}is1000a_Headset 1_mono.wav")
fs, audio3 = wavfile.read(f"{input_dir}is1000a_Headset 2_mono.wav")
fs, audio4 = wavfile.read(f"{input_dir}is1000a_Headset 3_mono.wav")
fs, bruit = wavfile.read(f"{input_dir}bruit.wav")

# Théorème de Sabine pour les paramètres de la méthode source-image :
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Création de la salle :
room = pra.ShoeBox(room_dim,
                   fs=fs,
                   materials=pra.Material(e_absorption),
                   max_order=max_order)

# Placement des sources dans la salle (les sources sont juxtaposées à l'aide du paramètre Delay) :

room.add_source([1, 2], signal=audio1, delay=0.5)
room.add_source([1, 4], signal=audio2, delay=5.5)
room.add_source([3, 2], signal=audio3, delay=17)
room.add_source([3, 4], signal=audio4, delay=24)
# Ajout d'un bruit de fond, d'une durée égale à la somme des 4 audios précédents
room.add_source([3.9, 5.9], signal=bruit[:(
    len(audio1)+len(audio2)+len(audio3)+len(audio4))], delay=0.5)


# Localisation/Création de l'antenne de micros :

Lg_t = 0.100                # Largeur du filtre (en s)
Lg = np.ceil(Lg_t*fs)       # en échantillons

center = [2, 3]
radius = 100e-3
fft_len = 512
echo = pra.circular_2D_array(center=center, M=8, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room.fs, N=fft_len, Lg=Lg)


# Placement de l'antenne :

room.add_microphone_array(mics)

# Affichage de la salle :

fig, ax = room.plot()

# Graphiques Beamforming :

room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 1")
room.mic_array.rake_delay_and_sum_weights(room.sources[1][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 2")
room.mic_array.rake_delay_and_sum_weights(room.sources[2][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 3")
room.mic_array.rake_delay_and_sum_weights(room.sources[3][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source 4")
room.mic_array.rake_delay_and_sum_weights(room.sources[4][:1])
fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
ax.legend(['500', '1000', '2000', '4000', '8000'])
plt.title("Source de bruit")

# Simulation (Construit la RIR automatiquement) :

room.simulate()

# Enregistrement du ignal audio reçu par l'antenne :
# Enregistrement du signal audio reçu par l'antenne :
name = "IS1000a_TR{:d}".format(int(rt60_tgt*1e3))
room.mic_array.to_wav(
    f"{output_dir}{name}.wav",
    norm=True,
    bitdepth=np.int16,
)

# Amélioration du résultat grâce au beamforming :
signal_das = mics.process(FD=False)
print("DAS Beamformed Signal:")
IPython.display.Audio(signal_das, rate=fs)

# Création du fichier texte regroupant les données de l'étude du beamforming (en cours...) :

# Permet d'afficher la matrice complète
np.set_printoptions(threshold=sys.maxsize)

# file = open("das.txt", "w") #Ouverture du fichier
# file.write(str(signal_das)) #ecriture
# file.close() #fermeture fichier


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
