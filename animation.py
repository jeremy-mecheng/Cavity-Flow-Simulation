import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#Input data
L = 100
U = np.zeros((2,L,L), dtype=float)
V = np.zeros((2,L,L), dtype=float)
R = np.zeros((2,L,L), dtype=float)

F = np.zeros((L,L), dtype=int)

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
inv = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

uo = 0.5

# Matrix Initialization
def init():
	for i in range(L):
		for j in range(L):
			U[0][i][j] = V[0][i][j] = 0
			U[1][i][j] = V[1][i][j] = 0
			R[0][i][j] = R[1][i][j] = 1
			F[i][j] = 0
			
			if j == 0 or i == 0 or i == L-1:
				F[i][j] = 1
			
			if j == L-1:
				U[0][i][j] = U[1][i][j] = uo

# Lattice Boltzman implementation			
def LBM(c):
	r, u, v, f = 0, 0, 0, 0
	for i in range(L):
		for j in range(L-1):
			if F[i][j] == 0:
				U[c][i][j] = V[c][i][j] = R[c][i][j] = 0
				for k in range(9):
					ip = i + ex[k]
					jp = j + ey[k]
					ik = inv[k]
					if F[ip][jp] == 0:
						r = R[1-c][ip][jp]
						u = U[1-c][ip][jp]/r
						v = V[1-c][ip][jp]/r
						f = w[ik]*r*(1-(3/2)*(u*u + v*v)+3*(ex[ik]*u+ey[ik]*v)+(9/2)*(ex[ik]*u+ey[ik]*v)*(ex[ik]*u + ey[ik]*v))
					else:
						f = w[ik]*R[1-c][i][j]
						
					R[c][i][j] += f
					U[c][i][j] += ex[ik]*f
					V[c][i][j] += ey[ik]*f

init()				

# --- Animation ---
fig, ax = plt.subplots()

def update(frame):
    c = frame % 2
    LBM(c)
    ax.clear()
    speed = np.sqrt(U[c]**2 + V[c]**2)
    im = ax.imshow(np.rot90(speed), cmap='viridis', interpolation='none')
    ax.set_title(f"Velocity field at step {frame}")
    return [im]

anim = FuncAnimation(fig, update, frames=501, interval=50, blit=False)

# --- Save as GIF ---
anim.save("Cavity_flow.gif", writer=PillowWriter(fps=20))