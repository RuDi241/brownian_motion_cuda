import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation

# Load data
dpi = 100
width = 21
height = width
cm = 1/2.54  # centimeters in inches
data = np.genfromtxt("data.txt")
n_particles = 1000
# Create a 3D plot
fig = plt.figure(figsize=(width*cm, height*cm), dpi=dpi)
ax = fig.add_subplot(111, projection='3d')

# Define a function to update the position of the particles each frame
def update(frame):
    ax.clear()
    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plot particles as points
    x = data[frame*n_particles:(frame+1)*n_particles, 1]
    y = data[frame*n_particles:(frame+1)*n_particles, 2]
    z = data[frame*n_particles:(frame+1)*n_particles, 3]
    ax.scatter(x, y, z, c='b', marker='o')
    # Make the first particle big and red
    ax.scatter(x[0], y[0], z[0], c='r', marker='o', s=100)
    ax.set_xlim([0, 1e4])
    ax.set_ylim([0, 1e4])
    ax.set_zlim([0, 1e4])
    ax.set_title("t = {:.2e} s".format(frame * 1e-12))



anim = animation.FuncAnimation(fig, update, frames=200, interval=100)

anim.save("anim.gif")

# Show plot
plt.show()
