import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Inputs
F_n = 100e-6  # Normal load in Newtons
a_0 = 300e-9  # contact radius in nm
A_0 = np.pi * a_0**2  # Contact area
sigma_0 = F_n / A_0  # 1 GPa
E_star = 70e9 # modulus in Pascals
S_dq_vals = [0.017, 0.007, 0.0065] # afm measurements of rms slope of the surfaces-SiOx, 1LG, 2LG

# Persson's erf formula: A_real/A_0 = erf( sigma_0 / (sqrt(2) * delta_p) )
# delta_p (rms pressure) = (E_star / 2) * S_dq
# So: A_real/A_0 = erf( sigma_0 / (sqrt(2) * (E_star / 2) * S_dq) )
# Simplified: erf( (sqrt(2) * sigma_0) / (E_star * S_dq) )

results = []
for sdq in S_dq_vals:
    argument = (np.sqrt(2) * sigma_0) / (E_star * sdq)
    ratio = erf(argument)
    results.append((sdq, argument, ratio))
plt.plot([r[0] for r in results], [r[2] for r in results], 'o-')
plt.xlabel('RMS Slope (S_dq)')
plt.ylabel('Contact Area Ratio (A_real/A_0)')
plt.title('Persson\'s Model')
plt.show()
print(results)
