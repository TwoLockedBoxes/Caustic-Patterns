import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

x, y, t = sp.symbols('x y t')

dim = (10, 10)  # Range of the plot, (+- x/2, +- y/2).
res = 25  # How detailed the surface is.
z_offset = 3  # How far above the xy-plane the surface is.
num_frames = 400  # Number of frames in the animation.

z = -sp.sin(sp.sqrt(x ** 2 + y ** 2) + t)  # The function defining the surface.

n1, n2 = 1, 1.33  # Refractive indices of the air and liquid medium, respectively. (For air and water, n1, n2 = 1, 1.33)

phi_0 = np.pi/24  # The angle of the incoming rays with respect to the z-axis.
phi_1 = 0  # The angle with respect to the x-axis.

t_vals = np.linspace(0, 6 * np.pi, num_frames)  # Set the endpoint to a multiple of 2*pi for smooth loops.

# Construct the normalized vector for the incoming parallel rays from parameters phi_0 and phi_1.
R_1_x = np.full(res, np.sin(phi_0) * np.cos(phi_1))
R_1_y = np.full(res, np.sin(phi_0) * np.sin(phi_1))
R_1_z = np.full(res, np.cos(phi_0))


# Generate outgoing rays based on surface normals and Snell's law.
def generate_vectors(func, time):

    x_vals = np.linspace(-dim[0]/2, dim[0]/2, res)
    y_vals = np.linspace(-dim[1]/2, dim[1]/2, res)
    X, Y = np.meshgrid(x_vals, y_vals)
    z = func + z_offset

    # Convert symbolic z function into a numerical function.
    z_num = sp.lambdify((x, y, t), z, 'numpy')
    Z = z_num(X, Y, time)

    dz_dx = sp.diff(z, x)
    dz_dy = sp.diff(z, y)
    dz_dx_num = sp.lambdify((x, y, t), dz_dx, 'numpy')
    dz_dy_num = sp.lambdify((x, y, t), dz_dy, 'numpy')

    '''Generating the basis (U, V) for the tangent plane at each point gives us two vectors. Taking the cross product
    of these vectors gives us a normal vector for the surface at each point. Working out this cross product, we find
    that the result of U x V is [dzdx, dzdy, 1]. We need to normalize it to use it as a basis for the outgoing ray.'''

    # The cross product of U and V.
    N_x = dz_dx_num(X, Y, time)
    N_y = dz_dy_num(X, Y, time)
    N_z = 1
    # Normalize.
    N_mag = np.sqrt(N_x**2 + N_y**2 + N_z**2)
    N_x = N_x / N_mag
    N_y = N_y / N_mag
    N_z = N_z / N_mag

    # Dot product of R_1 and N.
    R_1_dot_N = R_1_x * N_x + R_1_y * N_y + R_1_z * N_z
    Theta_1 = np.acos(R_1_dot_N)  # Angle of incoming ray with respect to normal vector.
    Theta_2 = np.asin((n1/n2) * np.sin(Theta_1))  # Angle of outgoing ray; calculated via Snell's law.
    Cos_Theta_2 = np.cos(Theta_2)

    '''We represent the outgoing ray in terms of a linear combination of R_1 and N, (aR_1 + bN). To find a and b, we
     use the constraints that ||R_2|| = 1, and R_2 dot N = cos(Theta_2). Solving these equations for b gives us the
      components of the outgoing ray in terms of R_1 and N.'''

    b = (Cos_Theta_2 - np.sqrt(R_1_dot_N**4 - R_1_dot_N**2 * Cos_Theta_2**2 + R_1_dot_N**2)) / (R_1_dot_N**2 + 1)
    a = np.sqrt(1 - b**2)

    # Generate the components of the outgoing ray, R_2.
    R_2_x = a * R_1_x + b * N_x
    R_2_y = a * R_1_y + b * N_y
    R_2_z = a * R_1_z + b * N_z

    # Normalize.
    R_2_mag = np.sqrt(R_2_x**2 + R_2_y**2 + R_2_z**2)
    R_2_x = R_2_x / R_2_mag
    R_2_y = R_2_y / R_2_mag
    R_2_z = R_2_z / R_2_mag

    '''We scale the normalized outgoing ray, R_2, by a constant, C, until it crosses the xy-plane. This occurs when the
    z-component of the vector is equal to the value of the function z(x, y) at that point. So the scaling factor C is
    determined. The x and y coordinates of the intersection are then given as the sum of the respective coordinates and
    their corresponding vector components in R_2.'''

    # Scale the vector R_2.
    C = Z / (R_2_z)
    R_2_x = R_2_x * C
    R_2_y = R_2_y * C

    # Calculate the intersection points.
    X_int = R_2_x + X
    Y_int = R_2_y + Y
    Z_int = np.zeros_like(X_int)

    return X, Y, Z, X_int, Y_int, Z_int


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-dim[0]/2, dim[0]/2)
ax.set_ylim(-dim[1]/2, dim[1]/2)
ax.set_zlim(0, z_offset)

X, Y, Z, Xs, Ys, Zs = generate_vectors(0)
surface = ax.plot_surface(X, Y, Z, cmap='viridis')
scatter = ax.scatter(Xs, Ys, Zs, color='blue', s=2)


def update(frame):
    global surface, scatter
    surface.remove()
    scatter.remove()

    t_val = t_vals[frame]
    X, Y, Z, Xs, Ys, Zs = generate_vectors(z, t_val)
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')
    scatter = ax.scatter(Xs, Ys, Zs, color='blue', s=2)
    return surface, scatter


ani = FuncAnimation(fig, update, frames=len(t_vals), interval=50, repeat=False)
ani.save("func_anim.mp4", writer='ffmpeg', fps=30)
print('done')

# Uncomment these lines to display the plot live.
#plt.tight_layout()
#plt.show()

