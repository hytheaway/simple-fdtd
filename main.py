import numpy as np
from scipy import signal
from scipy.io import wavfile

# --setup--

c = 343.0 # speed of sound
room = np.array([4.0, 3.0, 2.5]) # x, y, z

dx = 0.025 # grid spacing
# dt = 8.0e-5 # time step in s (~12.5kHz)
dt = 0.9 * dx / (c * np.sqrt(3))
fs = 1.0 / dt # fdtd sampling rate

T = 0.5 # total simulation time
Nt = int(T/dt)

# CFL
lambda_cfl = c * dt / dx

Nx = int(room[0] / dx)
Ny = int(room[1] / dx)
Nz = int(room[2] / dx)
nodes = Nx * Ny * Nz
print("grid size: ", Nx, Ny, Nz, "~>", nodes, "nodes")

# --allocate fields--

# pressure for previous, current, and next time steps
p_prev = np.zeros((Nx, Ny, Nz), dtype=np.float32)
p_curr = np.zeros_like(p_prev)
p_next = np.zeros_like(p_prev)

# room impulse response (receiver)
rir = np.zeros(Nt, dtype=np.float32)

# --source and receiver setup--

# source placed roughly 1/4 of room length, centered
i_s = Nx // 4
j_s = Ny // 2
k_s = Nz // 2

# receiver placed roughly 3/4 of room length, also centered
i_r = 3 * Nx // 4
j_r = Ny // 2
k_r = Nz // 2

# source signal
# # burst
# f0 = 500.0
# duration = 0.01 # in s
# Nb = int(duration * fs)
# n_b = np.arange(Nb)
# window = 0.5 - 0.5 * np.cos(2 * np.pi * n_b / (Nb - 1)) # hann window
# s = np.sin(2 * np.pi * f0 * n_b * dt) * window
# s = s.astype(np.float32)

# unit impulse
Nb = 1
s = np.zeros(1, dtype=np.float32)
s[0] = 1.0

# --main loop (fdtd)--

lam2 = (c * dt / dx) ** 2 # lambda^2

for n in range(Nt):
    if n < Nb:
        p_curr[i_s, j_s, k_s] += s[n]
    
    # update interior points w 7 point laplacian
    for i in range(1, Nx-1):
        # slice our neighbors along y,z for all interior j,k
        p_c = p_curr[i, 1:-1, 1:-1]
        lap = (
            p_curr[i+1, 1:-1, 1:-1] +
            p_curr[i-1, 1:-1, 1:-1] +
            p_curr[i, 2:, 1:-1] +
            p_curr[i, 0:-2, 1:-1] +
            p_curr[i, 1:-1, 2:] +
            p_curr[i, 1:-1, 0:-2] -
            6.0 * p_c
        )
        p_next[i, 1:-1, 1:-1] = 2.0 * p_c - p_prev[i, 1:-1, 1:-1] + lam2 * lap
    
    # boundary handling by copying edges from p_curr to be a crude approximation of rigid walls
    p_next[0, :, :] = p_curr[0, :, :]
    p_next[-1, :, :] = p_curr[-1, :, :]
    p_next[:, 0, :] = p_curr[:, 0, :]
    p_next[:, -1, :] = p_curr[:, -1, :]
    p_next[:, :, 0] = p_curr[:, :, 0]
    p_next[:, :, -1] = p_curr[:, :, -1]
    
    # receiver capture
    rir[n] = p_curr[i_r, j_r, k_r]
    
    # cycle buffers
    p_prev, p_curr, p_next = p_curr, p_next, p_prev
    
    if (n + 1) % 500 == 0:
        print(f"Step {n+1}/{Nt}")

# --save as .wav--

# wavfile.write("fdtd_rir.wav", int(fs), rir_audio.astype(np.float32))

# print("Wrote fdtd_rir.wav at fs=", fs)

# normalize
rir_max = np.max(np.abs(rir))
if rir_max > 0:
    rir_audio = rir / rir_max * 0.9
else:
    rir_audio = rir

# --resample--
target_fs = 48000

Nt_new = int(len(rir) * target_fs / fs)
rir_48k = signal.resample(rir, Nt_new)

# optional low-pass
f_cutoff = 1000.0
b, a = signal.butter(4, f_cutoff / (fs / 2), btype='low')
rir_lp = signal.filtfilt(b, a, rir)

rir_48k = signal.resample(rir_lp, Nt_new)

wavfile.write("fdtd_rir_48k.wav", target_fs, (rir_audio * 0.9).astype(np.float32))

print("Wrote fdtd_rir_48k.wav at fs=", target_fs)