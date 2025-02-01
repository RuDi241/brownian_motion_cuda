import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import subprocess

subprocess.run("nvcc main.cu".split())
runs = 10
diffusion_coefs = []
y_int = []
for i in range(runs):

    subprocess.run("./a.out".split())

    data = np.loadtxt("data.txt", skiprows=1)

    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]

    MSD = np.zeros(t.size)
    for i in range(1, t.size):
        MSD[i] = np.mean((x[:i] - x[0])**2 + (y[:i] - y[0])**2 + (z[:i] - z[0])**2)

    trs = int(3/4*len(t))
    trend = np.polyfit(t[trs:], MSD[trs:], 1)
    trendpoly = np.poly1d(trend)
    diffusion_coefs.append(trend[0]/6)
    y_int.append(trend[1])

with open('data.txt', 'r') as f:
    first_line = f.readline().strip()
    variables = first_line.split()
    i_temp = float(variables[0])
    num_of_particles = int(variables[1])
    time_step = float(variables[2])
    protein_mass = float(variables[3])
    protein_radius = float(variables[4])
    water_mass = float(variables[5])
    water_radius = float(variables[6])

mean_dif_coef = np.mean(diffusion_coefs)
mean_dev = np.std(diffusion_coefs)/np.sqrt(len(diffusion_coefs))
slope = mean_dif_coef*6
y_int_mean = np.mean(y_int)
line = lambda t:t*slope+y_int_mean
plt.plot(t, line(t))
plt.xlabel(r"$t(s)$")
plt.ylabel(r"$MSD(t)(A^2)$")
plt.title(f"protein_radius = {protein_radius} A, \nprotein_mass = {protein_mass} Da, T = {i_temp} K")
plt.legend([f'line => y = {slope:.3e}x + {y_int_mean:.3e} \nD = slope/(6) = {mean_dif_coef:.3e} +- {mean_dev:.2e} $A^2/s$'])
plt.savefig(f'protein_radius = {protein_radius} A, protein_mass = {protein_mass} Da, T = {i_temp} K, time step = {time_step:3e}s.')