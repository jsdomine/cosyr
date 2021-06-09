import numpy as np
import matplotlib.pyplot as plt


gamma = 100.0
alpha_max = 200.0/1e6*gamma**3.0
number_of_wavelets_pos = 20  # for positive alpha axis
number_of_wavelets_neg = 20   # for negative alpha axis

psi_max=(24.0*alpha_max/gamma**3.0)**(1.0/3.0)
psi_max_negative_axis = alpha_max/gamma**3.0/2.0

print(alpha_max, psi_max, psi_max_negative_axis)

psi_for_pos_axis = np.linspace(0.0, psi_max, number_of_wavelets_pos)
psi_for_neg_axis = np.linspace(psi_max_negative_axis, 0.0, number_of_wavelets_neg, endpoint=False)

alpha_axis = np.zeros(number_of_wavelets_pos + number_of_wavelets_neg)
alpha_axis[:number_of_wavelets_neg] = -psi_for_neg_axis*2.0
alpha_axis[number_of_wavelets_neg:] = psi_for_pos_axis**3.0/24.0

alpha_axis *= gamma**3.0


plt.plot(alpha_axis, "+")
