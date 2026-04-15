#!/usr/bin/env python
# coding: utf-8

# # Wigner Quasiprobability Distribution of Quantum Cat States
# 
# This notebook uses the `qutip` (Quantum Toolbox in Python) library to generate and plot the Wigner functions for an even and an odd quantum cat state.

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, wigner, destroy, position
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters ---
alpha = 3.0
N = 60

# --- 2. States ---
state_plus = coherent(N, alpha)
state_minus = coherent(N, -alpha)

even_cat = (state_plus + state_minus).unit()
odd_cat  = (state_plus - state_minus).unit()

rho_even = even_cat * even_cat.dag()
rho_odd  = odd_cat * odd_cat.dag()

# --- 3. Phase space ---
xvec = np.linspace(-6, 6, 300)
pvec = np.linspace(-6, 6, 300)

W_even = wigner(rho_even, xvec, pvec)
W_odd  = wigner(rho_odd,  xvec, pvec)


# In[16]:


# =========================
# 4. 2D WIGNER PLOTS
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

limit_even = np.max(np.abs(W_even))
axes[0].pcolormesh(xvec, pvec, W_even, cmap='RdBu',
                   vmin=-limit_even, vmax=limit_even, shading='auto')
axes[0].set_title('Even Cat ($\\alpha=3$)')
axes[0].set_xlabel('q'); axes[0].set_ylabel('p')
axes[0].set_aspect('equal')

limit_odd = np.max(np.abs(W_odd))
axes[1].pcolormesh(xvec, pvec, W_odd, cmap='RdBu',
                   vmin=-limit_odd, vmax=limit_odd, shading='auto')
axes[1].set_title('Odd Cat ($\\alpha=3$)')
axes[1].set_xlabel('q'); axes[1].set_ylabel('p')
axes[1].set_aspect('equal')

plt.subplots_adjust(wspace=0.3)
plt.show()


# In[18]:


# =========================
# 5. 3D WIGNER PLOTS
# =========================
X, P = np.meshgrid(xvec, pvec)

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, P, W_even, cmap='RdBu', edgecolor='none')
ax1.set_title('Even Cat (3D)', pad=15)
ax1.set_xlabel('q'); ax1.set_ylabel('p'); ax1.set_zlabel('W')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, P, W_odd, cmap='RdBu', edgecolor='none')
ax2.set_title('Odd Cat (3D)', pad=15)
ax2.set_xlabel('q'); ax2.set_ylabel('p'); ax2.set_zlabel('W')

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2)
plt.show()


# In[20]:


# =========================
# 6. MOMENTUM DISTRIBUTION
# =========================
a = destroy(N)

def momentum_distribution(rho, p_vals):
    probs = []
    for val in p_vals:
        ket = coherent(N, 1j*val/np.sqrt(2))
        proj = ket * ket.dag()
        probs.append((rho * proj).tr().real)
    return np.array(probs)

p = np.linspace(-6, 6, 300)
P_even = momentum_distribution(rho_even, p)
P_odd  = momentum_distribution(rho_odd, p)

P_even /= np.trapz(P_even, p)
P_odd  /= np.trapz(P_odd, p)

plt.figure(figsize=(10,5))
plt.plot(p, P_even, label="Even")
plt.plot(p, P_odd, label="Odd")
plt.xlabel("Momentum (p)")
plt.ylabel("Probability")
plt.title("Momentum Distribution")
plt.legend()
plt.show()


# In[22]:


# =========================
# 7. POSITION DISTRIBUTION
# =========================
x_op = position(N)
eigvals, eigvecs = x_op.eigenstates()

idx = np.argsort(eigvals)
x_vals = np.array(eigvals)[idx]
eigvecs = np.array(eigvecs)[idx]

P_x_even, P_x_odd = [], []

for ket in eigvecs:
    proj = ket * ket.dag()
    P_x_even.append((rho_even * proj).tr().real)
    P_x_odd.append((rho_odd * proj).tr().real)

P_x_even = np.array(P_x_even)
P_x_odd  = np.array(P_x_odd)

P_x_even /= np.trapz(P_x_even, x_vals)
P_x_odd  /= np.trapz(P_x_odd, x_vals)

plt.figure(figsize=(10,5))
plt.plot(x_vals, P_x_even, label="Even")
plt.plot(x_vals, P_x_odd, label="Odd")
plt.xlabel("Position (x)")
plt.ylabel("Probability")
plt.title("Position Distribution")
plt.legend()
plt.show()


# In[24]:


# =========================
# 8. INTERACTIVE 3D (PLOTLY)
# =========================
fig = make_subplots(rows=1, cols=2,
                   specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                   subplot_titles=("Even Cat", "Odd Cat"))

fig.add_trace(go.Surface(z=W_even, x=X, y=P, colorscale='RdBu'),
              row=1, col=1)

fig.add_trace(go.Surface(z=W_odd, x=X, y=P, colorscale='RdBu'),
              row=1, col=2)
fig.update_layout(
    title="Interactive Wigner Function",
    height=600, width=1100,
    scene=dict(
        xaxis_title='q',
        yaxis_title='p',
        zaxis_title='W(q,p)'
    ),
    scene2=dict(
        xaxis_title='q',
        yaxis_title='p',
        zaxis_title='W(q,p)'
    )
)

fig.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qutip import coherent, wigner, destroy

# --- Parameters ---
alpha = 3.0
N = 60
g = 1.0   # nonlinear strength
T = np.pi / (2*g)   # time for cat state formation

# Time steps
tlist = np.linspace(0, T, 240)

# Phase space grid
xvec = np.linspace(-6, 6, 200)
pvec = np.linspace(-6, 6, 200)

# --- Operators ---
a = destroy(N)
n_op = a.dag() * a

# Nonlinear Hamiltonian
H = g * (n_op * n_op)

# Initial coherent state
psi0 = coherent(N, alpha)

# --- Time evolution ---
states = []

for t in tlist:
    U = (-1j * H * t).expm()
    psi_t = U * psi0
    rho_t = psi_t * psi_t.dag()
    states.append(rho_t)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(6,5))

def update(frame):
    ax.clear()
    
    W = wigner(states[frame], xvec, pvec)
    limit = np.max(np.abs(W))
    
    cont = ax.contourf(xvec, pvec, W, 100,
                       cmap='RdBu',
                       vmin=-limit,
                       vmax=limit)
    
    ax.set_title(f"Time Evolution → Cat State (t = {tlist[frame]:.2f})")
    ax.set_xlabel("Position (q)")
    ax.set_ylabel("Momentum (p)")
    ax.set_aspect('equal')

# --- Animation ---
anim = FuncAnimation(fig, update, frames=len(tlist), interval=150)

plt.show()
anim.save("cat_state_evolution.gif", fps=5)


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, position

# --- Parameters ---
alpha = 3.0j
N = 60

# --- States ---
state_plus = coherent(N, alpha)
state_minus = coherent(N, -alpha)

# Bob's statistical mixture
rho_bob = 0.5 * (state_plus * state_plus.dag()) + \
          0.5 * (state_minus * state_minus.dag())

# --- Position operator ---
x_op = position(N)

# Eigen decomposition
eigvals, eigvecs = x_op.eigenstates()

# Sort eigenvalues
idx = np.argsort(eigvals)
x_vals = np.array(eigvals)[idx]
eigvecs = np.array(eigvecs)[idx]

# --- Compute probability distribution ---
P_x_bob = []

for ket in eigvecs:
    proj = ket * ket.dag()
    P_x_bob.append((rho_bob * proj).tr().real)

P_x_bob = np.array(P_x_bob)

# Normalize
P_x_bob /= np.trapz(P_x_bob, x_vals)

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(x_vals, P_x_bob, label="Bob (Statistical Mixture)")

plt.xlabel("Position (x)")
plt.ylabel("Probability Density")
plt.title("Position Distribution (Bob's Measurement)")
plt.legend()
plt.grid()

plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, position, momentum

# --- Parameters ---
alpha = 3.0j   # IMPORTANT: imaginary alpha (theory case)
N = 60

# --- States ---
state_plus  = coherent(N, alpha)
state_minus = coherent(N, -alpha)

# Alice (superposition)
psi_alice = (state_plus + state_minus).unit()
rho_alice = psi_alice * psi_alice.dag()

# Bob (statistical mixture)
rho_bob = 0.5 * (state_plus * state_plus.dag()) + \
          0.5 * (state_minus * state_minus.dag())

# --- Position Distribution ---
x_op = position(N)
eigvals_x, eigvecs_x = x_op.eigenstates()

# Sort
idx = np.argsort(eigvals_x)
x_vals = np.array(eigvals_x)[idx]
eigvecs_x = np.array(eigvecs_x)[idx]

P_x_alice = []
P_x_bob   = []

for ket in eigvecs_x:
    proj = ket * ket.dag()
    P_x_alice.append((rho_alice * proj).tr().real)
    P_x_bob.append((rho_bob * proj).tr().real)

P_x_alice = np.array(P_x_alice)
P_x_bob   = np.array(P_x_bob)

# Normalize
P_x_alice /= np.trapz(P_x_alice, x_vals)
P_x_bob   /= np.trapz(P_x_bob, x_vals)

# --- Momentum Distribution ---
p_op = momentum(N)
eigvals_p, eigvecs_p = p_op.eigenstates()

# Sort
idx = np.argsort(eigvals_p)
p_vals = np.array(eigvals_p)[idx]
eigvecs_p = np.array(eigvecs_p)[idx]

P_p_alice = []
P_p_bob   = []

for ket in eigvecs_p:
    proj = ket * ket.dag()
    P_p_alice.append((rho_alice * proj).tr().real)
    P_p_bob.append((rho_bob * proj).tr().real)

P_p_alice = np.array(P_p_alice)
P_p_bob   = np.array(P_p_bob)

# Normalize
P_p_alice /= np.trapz(P_p_alice, p_vals)
P_p_bob   /= np.trapz(P_p_bob, p_vals)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Position
axes[0].plot(x_vals, P_x_alice, label="Alice (Superposition)")
axes[0].plot(x_vals, P_x_bob, '--', label="Bob (Mixture)")
axes[0].set_title("Position Distribution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Probability Density")
axes[0].legend()
axes[0].grid()

# Momentum
axes[1].plot(p_vals, P_p_alice, label="Alice (Superposition)")
axes[1].plot(p_vals, P_p_bob, '--', label="Bob (Mixture)")
axes[1].set_title("Momentum Distribution")
axes[1].set_xlabel("p")
axes[1].set_ylabel("Probability Density")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig("comparison.png")
plt.show()

