"""Représentation interactive du repliement spectral (aliasing).

Ce script montre un signal sinusoïdal continu, ses échantillons pris
à une fréquence d'échantillonnage donnée, et la fréquence repliée
observée dans l'échantillonnage (aliasing).

Usage (exemples):
  python Repl_Spectral.py
  python Repl_Spectral.py --f_signal 8 --f_sample 10 --t_total 1 --save alias.png

Si le script est lancé sans arguments, il demandera les paramètres
interactivement.
"""

from typing import Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse


def generate_continuous_signal(frequency: float, duration: float, resolution: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
	"""Génère un signal sinusoïdal continu.

	Args:
		frequency: fréquence du signal en Hz.
		duration: durée en secondes.
		resolution: nombre de points pour la courbe continue.

	Returns:
		(t, y) arrays pour le temps et l'amplitude.
	"""
	t = np.linspace(0, duration, resolution)
	y = np.sin(2 * np.pi * frequency * t)
	return t, y


def sample_signal(frequency: float, sample_rate: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Échantillonne un signal sinusoïdal.

	Args:
		frequency: fréquence du signal en Hz.
		sample_rate: fréquence d'échantillonnage en Hz.
		duration: durée en secondes.

	Returns:
		(t_samples, y_samples)
	"""
	if sample_rate <= 0:
		raise ValueError("La fréquence d'échantillonnage doit être > 0")
	t_samples = np.arange(0, duration, 1.0 / sample_rate)
	y_samples = np.sin(2 * np.pi * frequency * t_samples)
	return t_samples, y_samples


def aliased_frequency(true_freq: float, sample_rate: float) -> float:
	"""Calcule la fréquence observée après repliement (aliasing).

	Cette formule ramène la fréquence dans l'intervalle [-Fs/2, Fs/2]
	et renvoie la valeur absolue (fréquence positive).
	"""
	if sample_rate <= 0:
		raise ValueError("La fréquence d'échantillonnage doit être > 0")
	# Ramener dans l'intervalle centré autour de 0
	ali = ((true_freq + sample_rate / 2) % sample_rate) - sample_rate / 2
	return abs(ali)


def plot_aliasing(f_signal: float, f_sample: float, t_total: float, save_path: str | None = None) -> None:
	"""Trace le signal continu, les échantillons et la composante repliée.

	Args:
		f_signal: fréquence du signal réel (Hz).
		f_sample: fréquence d'échantillonnage (Hz).
		t_total: durée en secondes.
		save_path: si fourni, enregistre la figure sous ce nom.
	"""
	t_cont, y_cont = generate_continuous_signal(f_signal, t_total, resolution=2000)
	t_samp, y_samp = sample_signal(f_signal, f_sample, t_total)

	ali_freq = aliased_frequency(f_signal, f_sample)
	# Estimer une phase pour que la courbe repliée s'aligne sur les échantillons
	# Modèle: y ~ A * sin(2π * f_alias * t + phi) => y ~ a*sin(wt) + b*cos(wt)
	w = 2 * np.pi * ali_freq
	if ali_freq == 0:
		# cas spécial: fréquence nulle -> constante (0)
		phi = 0.0
		amplitude = 0.0
	else:
		S = np.column_stack((np.sin(w * t_samp), np.cos(w * t_samp)))
		# résoudre en moindres carrés pour a,b
		coeffs, *_ = np.linalg.lstsq(S, y_samp, rcond=None)
		a, b = coeffs
		amplitude = np.hypot(a, b)
		phi = np.arctan2(b, a)

	# Générer la courbe repliée avec la phase estimée
	t_ali = np.linspace(0, t_total, 2000)
	y_ali = amplitude * np.sin(w * t_ali + phi)

	plt.figure(figsize=(10, 5))
	plt.plot(t_cont, y_cont, label=f'Vrai signal ({f_signal} Hz)', color='tab:blue', alpha=0.6)
	markerline, stemlines, baseline = plt.stem(t_samp, y_samp, linefmt='C1--', markerfmt='C1o', basefmt='k-')
	plt.setp(markerline, markersize=6)
	plt.plot(t_ali, y_ali, label=f'Signal replié observé ({ali_freq:.3f} Hz) — amplitude {amplitude:.3f}', color='tab:red', alpha=0.8)

	plt.title('Démonstration du repliement spectral (Aliasing)')
	plt.xlabel('Temps (secondes)')
	plt.ylabel('Amplitude')
	plt.legend()
	plt.grid(True)

	info_text = (
		f"Fs = {f_sample} Hz, Nyquist = {f_sample/2:.3f} Hz\n"
		f"Fréq. vraie = {f_signal} Hz, Fréq. observée (alias) = {ali_freq:.3f} Hz"
	)
	plt.gcf().text(0.01, 0.01, info_text, fontsize=9)

	if save_path:
		plt.savefig(save_path, dpi=150)
		print(f"Figure saved to {save_path}")

	plt.show()


def prompt_parameters() -> Tuple[float, float, float, str | None]:
	"""Invite l'utilisateur à entrer les paramètres (avec valeurs par défaut)."""
	def ask_float(prompt: str, default: float) -> float:
		while True:
			try:
				s = input(f"{prompt} [{default}]: ") or str(default)
				val = float(s)
				return val
			except ValueError:
				print("Valeur invalide — veuillez entrer un nombre.")

	f_signal = ask_float('Fréquence du signal (Hz)', 8.0)
	f_sample = ask_float('Fréquence d\'échantillonnage (Hz)', 10.0)
	t_total = ask_float('Durée totale (s)', 1.0)
	save = input('Nom du fichier pour sauvegarder la figure (laisser vide pour ne pas sauvegarder): ').strip() or None
	return f_signal, f_sample, t_total, save


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Démonstration du repliement spectral (aliasing)')
	parser.add_argument('--f_signal', type=float, help='Fréquence du signal réel (Hz)')
	parser.add_argument('--f_sample', type=float, help='Fréquence d\'échantillonnage (Hz)')
	parser.add_argument('--t_total', type=float, help='Durée totale (secondes)')
	parser.add_argument('--save', type=str, help='Chemin pour sauvegarder la figure (png, pdf, ... )')
	parser.add_argument('--interactive', action='store_true', help='Forcer le mode interactif (prompts)')
	parser.add_argument('--sliders', action='store_true', help='Lancer une interface graphique avec curseurs interactifs')
	return parser.parse_args()


def interactive_sliders(f_signal: float = 8.0, f_sample: float = 10.0, t_total: float = 1.0) -> None:
	"""Ouvre une fenêtre matplotlib avec deux curseurs pour ajuster
	`f_signal` et `f_sample` en temps réel.
	"""
	# préparation des données initiales
	t_cont, y_cont = generate_continuous_signal(f_signal, t_total, resolution=2000)
	t_samp, y_samp = sample_signal(f_signal, f_sample, t_total)

	fig, ax = plt.subplots(figsize=(10, 6))
	plt.subplots_adjust(left=0.1, bottom=0.25)

	# éléments initiaux
	cont_line, = ax.plot(t_cont, y_cont, color='tab:blue', alpha=0.6, label=f'Vrai signal ({f_signal} Hz)')
	stem = ax.stem(t_samp, y_samp, linefmt='C1--', markerfmt='C1o', basefmt='k-')
	markerline = stem[0]
	ali_line, = ax.plot([], [], color='tab:red', alpha=0.8, label='Signal replié')

	ax.set_title('Démonstration du repliement spectral (Aliasing)')
	ax.set_xlabel('Temps (secondes)')
	ax.set_ylabel('Amplitude')
	ax.grid(True)
	ax.legend()

	axcolor = 'lightgoldenrodyellow'
	ax_fsig = plt.axes([0.1, 0.12, 0.8, 0.03], facecolor=axcolor)
	ax_fsam = plt.axes([0.1, 0.07, 0.8, 0.03], facecolor=axcolor)

	s_fsig = Slider(ax_fsig, 'f_signal (Hz)', 0.0, max(1.0, f_sample * 4), valinit=f_signal)
	s_fsam = Slider(ax_fsam, 'f_sample (Hz)', 0.1, max(1.0, f_sample * 4), valinit=f_sample)

	resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
	button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

	def redraw(f_sig: float, f_sam: float) -> None:
		nonlocal stem
		# recalculer
		t_cont, y_cont = generate_continuous_signal(f_sig, t_total, resolution=2000)
		t_samp, y_samp = sample_signal(f_sig, f_sam, t_total)
		ali_freq = aliased_frequency(f_sig, f_sam)

		# estimer phase/amplitude pour aligner
		w = 2 * np.pi * ali_freq
		if ali_freq == 0:
			amplitude = 0.0
			phi = 0.0
		else:
			S = np.column_stack((np.sin(w * t_samp), np.cos(w * t_samp)))
			coeffs, *_ = np.linalg.lstsq(S, y_samp, rcond=None)
			a, b = coeffs
			amplitude = np.hypot(a, b)
			phi = np.arctan2(b, a)

		t_ali = np.linspace(0, t_total, 2000)
		y_ali = amplitude * np.sin(w * t_ali + phi)

		# mettre à jour les tracés
		cont_line.set_data(t_cont, y_cont)
		# mettre à jour stem: re-créer pour simplicité
		for coll in stem[1:]:
			try:
				coll.remove()
			except Exception:
				pass
		new_stem = ax.stem(t_samp, y_samp, linefmt='C1--', markerfmt='C1o', basefmt='k-')
		# remplacer les références
		stem = new_stem
		ali_line.set_data(t_ali, y_ali)

		ax.relim()
		ax.autoscale_view()
		info_text = f"Fs = {f_sam:.3f} Hz, Nyquist = {f_sam/2:.3f} Hz\nFréq vraie = {f_sig:.3f} Hz, Aliased = {ali_freq:.3f} Hz"
		# retirer ancien texte si existe
		for txt in ax.texts:
			txt.remove()
		ax.text(0.01, 0.01, info_text, fontsize=9, transform=fig.transFigure)
		fig.canvas.draw_idle()

	def update(val):
		redraw(s_fsig.val, s_fsam.val)

	def reset(event):
		s_fsig.reset()
		s_fsam.reset()

	s_fsig.on_changed(update)
	s_fsam.on_changed(update)
	button.on_clicked(reset)

	# trace initial
	redraw(f_signal, f_sample)
	plt.show()


def main() -> None:
	args = parse_args()

	if args.interactive or (not any([args.f_signal, args.f_sample, args.t_total, args.save]) and sys.stdin.isatty()):
		f_signal, f_sample, t_total, save = prompt_parameters()
	else:
		# Utiliser les valeurs fournies ou défauts
		f_signal = args.f_signal if args.f_signal is not None else 8.0
		f_sample = args.f_sample if args.f_sample is not None else 10.0
		t_total = args.t_total if args.t_total is not None else 1.0
		save = args.save

	try:
		plot_aliasing(f_signal, f_sample, t_total, save_path=save)
	except Exception as e:
		print(f"Erreur: {e}")


if __name__ == '__main__':
	main()