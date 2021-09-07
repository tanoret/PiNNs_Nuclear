import sys
sys.path.insert(0, '../Utils/')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pinn_NS import PhysicsInformedNN_NS
from logger import Logger
from lbfgs import Struct

#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 5000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 1000 # Number of samples per boundary
N_internal = 20000 # Number of samples per boundary

#############################
# Network srchitecture
#############################
layers = [2] + 8*[20] + [2]

#################################
# Setting logger and optimizer
#################################
logger = Logger(frequency=20)
def error():
  u_pred, v_pred, p_pred = pinn_velocity.predict(X_u=np.concatenate([[points_dict['x_bc'], points_dict['y_bc']]], 1).T)
  return np.linalg.norm(bc_dict['u_bc'] - u_pred, 2) / np.linalg.norm(bc_dict['u_bc'], 2)
logger.set_error_fn(error)

#################################
# Setting up tf optimizer
#################################
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

coupled_optimizer = {}
coupled_optimizer['nt_config'] = Struct()
coupled_optimizer['nt_config'].learningRate = 0.9
coupled_optimizer['nt_config'].maxIter = nt_epochs
coupled_optimizer['nt_config'].nCorrection = 50
coupled_optimizer['nt_config'].tolFun = 1.0 * np.finfo(float).eps
coupled_optimizer['batches'] = n_batches


###########################################
# Setting up internal and boundary points
###########################################

# Domain
x_min, x_max = 0., 2.
y_min, y_max = 0., 2.

# Boundaries
bottom = (np.linspace(x_min,x_max,N_boundary), np.linspace(y_min,y_min,N_boundary))
top    = (np.linspace(x_min,x_max,N_boundary), np.linspace(y_max,y_max,N_boundary))
left   = (np.linspace(x_min,x_min,N_boundary), np.linspace(y_min,y_max,N_boundary))
right  = (np.linspace(x_max,x_max,N_boundary), np.linspace(y_min,y_max,N_boundary))

x_cord_bc = np.array([bottom[0], top[0], left[0], right[0]])
y_cord_bc = np.array([bottom[1], top[1], left[1], right[1]])

# Boundary conditions
u_bc = np.array([bottom[0]*0., top[0]*0.+1., left[0]*0., right[0]*0.])
v_bc = np.array([bottom[1]*0., top[1]*0.,    left[1]*0., right[1]*0.])

# Internal points
n_points = int(np.sqrt(N_internal))
x_max, y_max = 2, 2
delta_x, delta_y = x_max/n_points, y_max/n_points
nx, ny = (n_points, n_points)
nd = 4.0
x = np.linspace(0 + delta_x/nd, x_max - delta_x/nd, nx)
y = np.linspace(0 + delta_x/nd, y_max - delta_x/nd, ny)
x_grid, y_grid = np.meshgrid(x, y)
x_cord_int, y_cord_int = x_grid, y_grid

# Creating dictionary of internal points
points_dict = {}
points_dict['x_eq'] = x_cord_int.flatten()
points_dict['y_eq'] = y_cord_int.flatten()
points_dict['x_bc'] = x_cord_bc.flatten()
points_dict['y_bc'] = y_cord_bc.flatten()

# Creating boundary conditions dictionary
bc_dict = {}
bc_dict['u_bc'] = u_bc.flatten()
bc_dict['v_bc'] = v_bc.flatten()


#########################################
# Defining thermophysical properties
#########################################
tp_prop = {'nu': 0.025, 
           'alpha': 1e-4, #9.1e-8, 
           'T_ref': 900, 
           'g': np.array([0, -9.81]), 
           'Cv': 6.15e6, 
           'alpha_c': 1e-4, #1.25e-10,
           'therm_dil': 2e-4}

#########################################
# Trainign the PiNNs
#########################################
# Creating the PiNNs model
pinn_velocity = PhysicsInformedNN_NS(layers=layers, optimizer=tf_optimizer, logger=logger, 
                            points_dict=points_dict, bc_dict=bc_dict, tp_properties=tp_prop)

# Training
pinn_velocity.fit(tf_epochs, coupled_optimizer, restart_tf = False)


#########################################
# Plotting
#########################################
nx, ny = (20, 20)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
x_grid, y_grid = np.meshgrid(x, y)
x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
X_u = np.concatenate([[x_grid, y_grid]], 1).T
u, v, p = pinn_velocity.predict(X_u)

plt.figure(figsize=[5,5])
pl = plt.tricontourf(x_grid, y_grid, u, levels=30)
plt.colorbar(pl)
plt.title('U velocity')

plt.figure(figsize=[5,5])
pl = plt.tricontourf(x_grid, y_grid, v, levels=30)
plt.colorbar(pl)
plt.title('V velocity')

plt.figure(figsize=[5,5])
pl = plt.tricontourf(x_grid, y_grid, p, levels=30)
plt.colorbar(pl)
plt.title('Pressure')

plt.figure(figsize=[5,5])
pl = plt.tricontourf(x_grid, y_grid, np.sqrt(u**2 + v**2), levels=30)
plt.colorbar(pl)
plt.title('Velocity Magnitude')