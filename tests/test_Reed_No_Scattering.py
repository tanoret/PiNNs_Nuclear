import sys
sys.path.insert(0, '../Utils/')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pinn_ADR import PhysicsInformedNN_ADR
from logger import Logger
from lbfgs import Struct

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)

#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 100 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]


##############################
# Parameters
##############################
# # Reference
sigt  = np.array([50., 5., 0., 1.,  1. ])
sigs  = np.array([ 0., 0., 0., 0.9, 0.9])
qext  = np.array([50., 0., 0., 1. , 0. ])
width = np.array([ 2., 1., 2., 1. , 2. ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 8.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=20)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

#################################
# Setting up tf optimizer
#################################
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

coupled_optimizer = {}
coupled_optimizer['nt_config'] = Struct()
coupled_optimizer['nt_config'].learningRate = 0.5
coupled_optimizer['nt_config'].maxIter = nt_epochs
coupled_optimizer['nt_config'].nCorrection = 50
coupled_optimizer['nt_config'].tolFun = 1.0 * np.finfo(float).eps
coupled_optimizer['batches'] = n_batches


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = qext[2]
  power[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = qext[3]
  power[(x_mesh >= agg_width[3])] = qext[4]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = sigt[2]
  self_reaction_coef[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = sigt[3]
  self_reaction_coef[(x_mesh >= agg_width[3])] = sigt[4]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

for i, _ in enumerate(mu_q):
  dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)

scalar_flux = compute_scalar_flux(dict_pinns)