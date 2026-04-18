# Generated from: Decoherence of Cat States.ipynb
# Converted at: 2026-04-18T08:40:21.789Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation

# --- PARAMETERS ---
N = 50
alpha = 3.0
gamma = 0.2        # decoherence rate
omega = 1.0

# --- OPERATORS ---
a = destroy(N)
H = omega * a.dag() * a

# --- INITIAL CAT STATE ---
psi_plus = coherent(N, alpha)
psi_minus = coherent(N, -alpha)

psi0 = (psi_plus + psi_minus).unit()   # even cat
rho0 = ket2dm(psi0)

# --- TIME GRID ---
tlist = np.linspace(0, 5, 100)

# --- COLLAPSE OPERATOR (decoherence) ---
c_ops = [np.sqrt(gamma) * a]

# --- SOLVE MASTER EQUATION ---
result = mesolve(H, rho0, tlist, c_ops, [])

# --- PHASE SPACE GRID ---
xvec = np.linspace(-6, 6, 200)

# --- SETUP PLOT ---
fig, ax = plt.subplots(figsize=(6,5))

def update(frame):
    ax.clear()
    
    rho_t = result.states[frame]
    W = wigner(rho_t, xvec, xvec)
    
    cont = ax.contourf(xvec, xvec, W, 100, cmap='RdBu')
    
    ax.set_title(f"Decoherence of Cat State by decaying of amplitude (t = {tlist[frame]:.2f})")
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_aspect('equal')

# --- ANIMATION ---
ani = FuncAnimation(fig, update, frames=len(tlist), interval=300)
ani.save("cat_decoherence_decay.gif", fps=3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation

# --- PARAMETERS ---
N = 50
alpha = 3.0
gamma = 0.2        # dephasing rate
omega = 1.0

# --- OPERATORS ---
a = destroy(N)
H = omega * a.dag() * a

# --- INITIAL CAT STATE ---
psi_plus = coherent(N, alpha)
psi_minus = coherent(N, -alpha)

psi0 = (psi_plus + psi_minus).unit()   # even cat
rho0 = ket2dm(psi0)

# --- TIME GRID ---
tlist = np.linspace(0, 5, 100)

# --- COLLAPSE OPERATOR (dephasing) ---
c_ops = [np.sqrt(gamma) * a.dag() *a]

# --- SOLVE MASTER EQUATION ---
result = mesolve(H, rho0, tlist, c_ops, [])

# --- PHASE SPACE GRID ---
xvec = np.linspace(-6, 6, 200)

# --- SETUP PLOT ---
fig, ax = plt.subplots(figsize=(6,5))

def update(frame):
    ax.clear()
    
    rho_t = result.states[frame]
    W = wigner(rho_t, xvec, xvec)
    
    cont = ax.contourf(xvec, xvec, W, 100, cmap='RdBu')
    
    ax.set_title(f"Decoherence of Cat State by dephasing (t = {tlist[frame]:.2f})")
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_aspect('equal')

# --- ANIMATION ---
ani = FuncAnimation(fig, update, frames=len(tlist), interval=300)
ani.save("cat_decoherence_dephasing.gif", fps=3)
plt.show()