#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import wave
import math
import threading as th
import time

import numpy as np
import scipy.fft, scipy.signal
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.widgets as wid
import matplotlib.gridspec as grid
from playsound import playsound

from utils.wav import *
from utils.signals import *
from utils.fft import *


SAMPLING_FREQ = 44100


def build_draw_frame_fn(refresh_period_ms):
	last_t = 0
	# Fonction qui se fait appelée périodiquement pour redessiner le graphique
	def draw_frame(frame, fig, graph, line, spec):
		global playing
		nonlocal last_t
		if not playing:
			fig.canvas.draw()
			fig.canvas.flush_events()
			return

		# On met dans la variable x l'axe fréquentiel et dans y l'axe de valeurs de la prochaine itération du spectrogramme.
		try:
			y, x = next(spec)
		# S'il ne reste rien à traiter, on met des listes vides dans x et y (ça va juste rien faire).
		except StopIteration:
			x, y = [], []

		# On met à jour seulement les données des lignes (avec nos deux axes) et on redessine le graphique.
		line.set_xdata(x)
		line.set_ydata(y)
		fig.canvas.draw()
		fig.canvas.flush_events()

		# On attend jusqu'au prochain rafraichissement déterminé par refresh_period_ms
		while (time.time_ns() - last_t)/1_000_000 < refresh_period_ms:
			pass
		print(frame, (time.time_ns()-last_t)/1_000_000)
		last_t = time.time_ns()
	return draw_frame

def build_spectrogram_animation(filename, fft_size, x_range=None, y_range=None):
	# On charge le fichier, le mixer (en normalisant) et créer son spectrogramme.
	channels, fps = load_wav(filename)
	sig = mix_signals(channels, 0.89)
	# On crée le spectrogramme. On utilse une fenêtre de Hanning (on passe "hann")
	spec = spectrogram(sig, fft_size, fps, "hann")

	# Création de la figure en laissant de l'espace en bas pour des boutons (ou autres)
	fig = plt.figure("Spectrogram")
	gs = grid.GridSpec(2, 1, height_ratios=(6, 1), figure=fig)

	# Création du graphe dans l'espace du haut.
	graph = fig.add_subplot(gs[0, 0])
	# On applique une échelle logarithmique à l'axe des X.
	graph.set_xscale("log")
	# On contraint les valeurs des axes si `x_range` ou `y_range` ne sont pas None.
	if x_range is not None:
		graph.set_xlim(*x_range)
	if y_range is not None:
		graph.set_ylim(*y_range)

	# Création de la courbe qui va dessiner la FFT.
	line = graph.plot([], [])[0]

	refresh_period_ms = 1000 / (fps / fft_size)
	print(f"expected: {refresh_period_ms} ms")

	last_t = 0
	# Fonction qui se fait appelée périodiquement pour redessiner le graphique
	draw_frame = build_draw_frame_fn(refresh_period_ms)

	return fig, anim.FuncAnimation(fig, draw_frame, fargs=(fig, graph, line, spec), interval=1)

def wait_and_play(filename):
	global playing
	while not playing:
		# On fait une pause pour relâcher le contrôle sur le processus.
		time.sleep(0.01)
	playsound(filename, block=False)


playing = False # Contrôle le départ du dessin et de la musique.


def main():
	try:
		os.mkdir("output")
	except:
		pass

	set_signal_gen_sampling_rate(SAMPLING_FREQ)

	# Un accord majeur (racine, tierce, quinte, octave) en intonation juste
	root_freq = 220
	root = sine(root_freq, 1, 2.0)
	third = sine(root_freq * 5/4, 1, 2.0)
	fifth = sine(root_freq * 3/2, 1, 2.0)
	octave = sine(root_freq * 2, 1, 2.0)
	notes = (root, third, fifth, octave)
	# On plaque et on arpège.
	block_chord = normalize(root + third + fifth + octave, 0.89)
	arpeggio = normalize(np.concatenate([e[:len(e)//2] for e in notes]), 0.89)

	save_wav(block_chord, "output/major_chord.wav", 1, SAMPLING_FREQ)
	save_wav(arpeggio, "output/major_chord_arpeggio.wav", 1, SAMPLING_FREQ)

	# TODO: Afficher la FFT de `block_chord` dans une fenêtre.
	
	# TODO: Pour chaque note générée précédemment (dans `notes`), afficher sa FFT. On veut ici les afficher indépendamment, mais sur le même graphique.

	wav_filename = "data/stravinsky.wav"

	# Création de l'animation. On contraint ici nos axes pour visionner le domaine intéressant des données.
	fig, ani = build_spectrogram_animation(wav_filename, 4096, (20, 10_000), (0, 0.2))
	
	# Création du bouton qui part le dessin et la musique.
	btn_pos = fig.add_axes([0.8, 0.05, 0.15, 0.10])
	def start_play(event):
		global playing
		playing = True
	btn = wid.Button(btn_pos, "START")
	btn.on_clicked(start_play)

	# Création du thread qui part la musique quand on appuie sur le bouton.
	p = th.Thread(target=wait_and_play, args=(wav_filename,))
	p.start()

	plt.show()
	p.join()

if __name__ == "__main__":
	main()

