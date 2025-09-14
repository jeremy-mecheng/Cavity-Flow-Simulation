import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

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

# Function to update the figure in every step
def update(frame):
    c = frame % 2  # Vary between 0 and 1
    LBM(c)  # Call LBM to update velocity distribution
    
    # Clean previous figure
    plt.clf()
    
    # Update the visualization
    plt.subplot(1, 1, 1)  # Main subplot
    speed = np.sqrt(U[c]**2 + V[c]**2)  # Velocity magnitude
    speed_r = np.rot90(speed)
    plt.imshow(speed_r, cmap='viridis', interpolation='none')
    plt.colorbar(label='Magnitud de la Velocidad')
    plt.title(f'Evolución de la Velocidad en el Paso {frame}')
    plt.xlabel('X')
    plt.ylabel('Y')

# Create figure for animation
fig = plt.figure(figsize=(6, 6))

# Create animation
anim = ani.FuncAnimation(fig, update, frames=101, interval=50)  # 1001 steps, interval of 50 ms between frames

# Show animation
plt.show()
