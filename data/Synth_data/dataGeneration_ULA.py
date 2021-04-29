import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import IPython
import sys
import os
import wave
import scipy.signal as dsp
import pyroomacoustics.datasets.cmu_arctic as dataset 


"""
Parameters
"""
### Repositories
dataset="cmu"
# CMU ARCTIC dataset
base_dir = "data/Synth_data/input/CMU/"
spk_list = [
    {"cluster":"aew","sentence":1},
    {"cluster":"axb","sentence":55},
]
# Where simulation results are stored
output_dir = "data/Synth_data/output/ULA/"


### Simulation parameters
# Target sampling rate [Hz] (resampling if needed)
fs = 16000
# RT60 [s]
rt60_tgt = 0.3  # en secondes
room_dim = [4, 6]  # en mètres
# Signal to noise ratio
snr = 10
ref_mic_idx = 0
# if True, add noise source in a corner of the room in addition to noise added by simulation
noise_src_fl = False
noise_src_loc = [3.9, 5.9]

### Sources DOAs [deg]
doa_src = [45,125]
# overlap between the sources above
#overlap_list = [5.0]
overlap_list = [1.0,4.0,2.0]
# distance from array center [m]
src_dist = 1.

### microphone array properties
nb_mic = 4
x_start = -0.1
x_stop = 0.1

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
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Importation des signaux audios (up to 4):
nb_src = len(doa_src)
doa_src = np.array(doa_src,dtype=np.float)
i=0
audio = []
while i < nb_src:
    audio_name = "{}cmu_us_{}_arctic/wav/arctic_a{:04d}.wav".format(base_dir,
                                                                  spk_list[i]["cluster"],
                                                                  spk_list[i]["sentence"])
    fs_audio, sig = wavfile.read(audio_name)

    # adjust sampling rate if audio does not fit targeted one
    if fs_audio > fs:
        sig = dsp.decimate(x=sig,q=fs_audio//fs)  

    audio.append(sig)
    i+=1    


# Théorème de Sabine pour les paramètres de la méthode source-image :
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Création de la salle :
room = pra.ShoeBox(room_dim,
                   fs=fs,
                   materials=pra.Material(e_absorption),
                   max_order=max_order)


# Source positions
doa_src*=np.pi/180.
srcpnts = np.zeros((nb_src, 2))
start = 0.1
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
    nb_src+=1



# create linear microphone array
x_step = (x_stop-x_start)/nb_mic
x_mic = np.arange(x_start,x_stop,x_step)
y_mic = np.zeros(x_mic.shape)
micropnts = np.array([x_mic,y_mic]).T
micropnts += center

mics = pra.beamforming.Beamformer(np.array(micropnts).T, fs, N=fft_len, Lg=Lg, hop=None)

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
    room.simulate(reference_mic=ref_mic_idx,snr=snr)
else:
    room.simulate()
    snr="inf"

"""
Saving and post processing
"""
# Enregistrement du signal audio reçu par l'antenne :
if dataset == "ami":
    name = "IS1000a_TR{:d}_T{:d}_nch{:d}_snr{}_ola{:d}_noise{:d}".format(int(rt60_tgt*1e3),
                                                                        int(np.ceil(
                                                                            total_duration)),
                                                                        int(nb_mic),
                                                                        str(snr),
                                                                        ola_ratio,
                                                                        int(noise_src_fl))
elif dataset == "cmu":
    name = "{}{}_TR{:d}_T{:d}_nch{:d}_snr{}_ola{:d}_noise{:d}".format(spk_list[0]["cluster"],
                                                                        spk_list[1]["cluster"],
                                                                        int(rt60_tgt*1e3),
                                                                        int(np.ceil(
                                                                            total_duration)),
                                                                        int(nb_mic),
                                                                        str(snr),
                                                                        ola_ratio,
                                                                        int(noise_src_fl))

room.mic_array.to_wav(
    f"{output_dir}{name}.wav",
    norm=True,
    bitdepth=np.int16)

# Amélioration du résultat grâce au beamforming :
signal_das = mics.process(FD=False)
signal_das/= np.max(np.abs(signal_das))
# Enregistrement du ignal audio reçu par l'antenne :
name = "BF_"+name
wavfile.write(filename=f"{output_dir}{name}.wav",
              rate=fs,
              data=signal_das)

#TODO Création du fichier texte regroupant les données de l'étude du beamforming (en cours...) :

# what is that ?
#IPython.display.Audio(signal_das, rate=fs)
# Permet d'afficher la matrice complète
#np.set_printoptions(threshold=sys.maxsize)


# Affiche le nombre de canal du signal ainsi que le nombre d'échantillons
print(f"Signal shape : {room.mic_array.signals.shape}")

# Mesure du TR60 :

rt60 = room.measure_rt60()
print("Le TR60 désiré est {}".format(rt60_tgt))  # TR60 choisi
print("Le TR60 mesuré est {}".format(rt60[1, 0]))  # TR60 mesuré