import numpy as np
import matplotlib.pyplot as plt

# Paramètres
F_signal = 8.0  # Fréquence du signal réel (Hz)
F_e = 10.0      # Fréquence d'échantillonnage (Hz)
temps_total = 1.0

# Temps continu (haute résolution pour dessiner la vraie courbe)
t_continu = np.linspace(0, temps_total, 1000)
signal_vrai = np.sin(2 * np.pi * F_signal * t_continu)

# Points d'échantillonnage (les "photos" prises par le système)
t_echantillons = np.arange(0, temps_total, 1/F_e)
echantillons = np.sin(2 * np.pi * F_signal * t_echantillons)

# Affichage
plt.figure(figsize=(10, 5))
plt.plot(t_continu, signal_vrai, label=f'Vrai signal ({F_signal} Hz)', color='blue', alpha=0.5)
plt.stem(t_echantillons, echantillons, linefmt='r--', markerfmt='ro', label=f'Échantillons (pris à {F_e} Hz)')
plt.plot(t_continu, np.sin(2 * np.pi * (F_e - F_signal) * t_continu), label=f'Signal fantôme reconstruit ({(F_e - F_signal)} Hz)', color='red')

plt.title("Démonstration du repliement spectral (Aliasing)")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()