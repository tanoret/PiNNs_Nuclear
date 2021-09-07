import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from fourier_projection import FourierFeatureProjection
from lbfgs import lbfgs
from lbfgs import Struct


class PhysicsInformedNN_NS(object):
  def __init__(self, 
               layers, 
               optimizer, 
               logger, 
               points_dict,
               bc_dict,
               tp_properties):
    
      
    # Setting up dimension and handling colocation and boundary points
    # So far only dimension 2 is supported
    
    # Internal points
    x_eq, y_eq = points_dict['x_eq'], points_dict['y_eq']
    x_bc, y_bc = points_dict['x_bc'], points_dict['y_bc']
    X_f = np.concatenate([[x_bc, y_bc]], 1).T
    X_eq = np.concatenate([[x_eq, y_eq]], 1).T
    
    # Boundary points
    u_bc, v_bc = bc_dict['u_bc'], bc_dict['v_bc']
    U_f = np.concatenate([[u_bc, v_bc]], 1).T

    # Crating PiNN model
    ub, lb = X_f.max(), X_f.min()
    self.ub, self.lb = ub, lb

    # Descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    # Assign logegr and optimizer
    self.optimizer = optimizer
    self.logger = logger

    self.dtype = tf.float32

    # Separating the collocation coordinates
    self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
    self.y_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)
    self.x_eq = tf.convert_to_tensor(X_eq[:, 0:1], dtype=self.dtype)
    self.y_eq = tf.convert_to_tensor(X_eq[:, 1:2], dtype=self.dtype)
    self.u_f = tf.convert_to_tensor(U_f[:, 0:1], dtype=self.dtype)[..., 0]
    self.v_f = tf.convert_to_tensor(U_f[:, 1:2], dtype=self.dtype)[..., 0]
    self.X_f = X_f

    # Initialize Lagrange multipliers for homogeneous penalty
    self.lambda_bc, self.lambda_u, self.lambda_v = 1.,  1.,  1.
    self.scaling_penalty = len(x_eq)**2
    
    ###########################################
    # Physics part
    ###########################################

    # Assign thermophysical properties
    self.nu = tp_properties['nu']
    self.alpha = tp_properties['alpha']
    self.T_ref = tp_properties['T_ref']
    self.g = tp_properties['g']

    # Initialize coupled fields
    self.T = tf.convert_to_tensor(X_eq[:, 0:1] * 0. + tp_properties['T_ref'], dtype=self.dtype)[..., 0] # Initializing temperature field in zero
    self.T_ref = tf.convert_to_tensor(X_eq[:, 0:1] * 0. + tp_properties['T_ref'], dtype=self.dtype)[..., 0] 
    

  def __loss(self, lambda_bc = False, lambda_u = False, lambda_v = False):
    _, _, _, f_u_pred, f_v_pred, _ = self.f_model()
    u_pred, v_pred, p = self.f_model_predict()
    lambda_bc = lambda_bc if lambda_bc else self.lambda_bc
    lambda_u  = lambda_u  if lambda_u  else self.lambda_u
    lambda_v  = lambda_v  if lambda_v  else self.lambda_v
    return  lambda_bc * tf.reduce_sum(tf.square(self.u_f - u_pred)) + \
            lambda_bc * tf.reduce_sum(tf.square(self.v_f - v_pred)) + \
            lambda_bc * tf.abs(p[...,0][0] - tf.constant(0.)) * 0.+\
            lambda_u * tf.reduce_sum(tf.square(f_u_pred)) + \
            lambda_v * tf.reduce_sum(tf.square(f_v_pred)) + \
            tf.constant(self.scaling_penalty, dtype=self.dtype) * \
            tf.square(
               tf.constant(lambda_bc, dtype=self.dtype) + \
               tf.constant(lambda_u, dtype=self.dtype) + \
               tf.constant(lambda_v, dtype=self.dtype) - \
               tf.constant(3.0, dtype=self.dtype)
              )
            #tf.reduce_sum(tf.square(f_c_pred)) + \ #Continuity loss
            

  def __grad(self):
    with tf.GradientTape() as tape:
      loss_value = self.__loss()
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var

  def update_coupling(self, T):
    self.T = tf.convert_to_tensor(T, dtype=self.dtype)

  ###########################################
  # Physics part
  ###########################################
  # The actual PINN
  def f_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    
    with tf.GradientTape(persistent=True) as tape:

      # Watching the two inputs we’ll need later, x and t
      tape.watch(self.x_eq)
      tape.watch(self.y_eq)

      # Packing together the inputs
      X_eq = tf.stack([self.x_eq[:,0], self.y_eq[:,0]], axis=1)

      # Getting the prediction
      psi_and_p = self.u_model(X_eq)
      psi = psi_and_p[..., 0, tf.newaxis]
      p = psi_and_p[..., 1, tf.newaxis]

      # Getting velocities
      u = tape.gradient(psi, self.y_eq)[..., 0]
      v = -tape.gradient(psi, self.x_eq)[..., 0]

      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, self.x_eq)[..., 0]
      u_y = tape.gradient(u, self.y_eq)[..., 0]
      v_x = tape.gradient(v, self.x_eq)[..., 0]
      v_y = tape.gradient(u, self.y_eq)[..., 0]
    
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, self.x_eq)[..., 0]
    u_yy = tape.gradient(u_y, self.y_eq)[..., 0]
    
    v_xx = tape.gradient(v_x, self.x_eq)[..., 0]
    v_yy = tape.gradient(v_y, self.y_eq)[..., 0]
    
    p_x = tape.gradient(p, self.x_eq)[..., 0]
    p_y = tape.gradient(p, self.y_eq)[..., 0]

    # Letting the tape go
    del tape

    nu = self.get_params(numpy=True)

    f_u = (u*u_x + v*u_y) + p_x - nu*(u_xx + u_yy) + self.g[0]*self.alpha*(self.T - self.T_ref)
    f_v = (u*v_x + v*v_y) + p_y - nu*(v_xx + v_yy) + self.g[1]*self.alpha*(self.T - self.T_ref)
    f_c = u_x + u_y

    # Buidling the PINNs
    return u, v, p, f_u, f_v, f_c


  def get_params(self, numpy=False):
    return self.nu

  def get_weights(self):
    w = []
    for layer in self.u_model.layers[1:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.u_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def summary(self):
    return self.u_model.summary()

  def return_numpy_loss(self, lambda_array):
    lambda_bc, lambda_u, lambda_v = lambda_array
    return self.__loss(lambda_bc, lambda_u, lambda_v).numpy()

  def __return_jacobian(self, lambda_array):
    return ndiff.Jacobian(lambda x: self.return_numpy_loss(x))(lambda_array).ravel()

  # The training function
  def fit(self, tf_epochs=5000, coupled_optimizer={'nt_config': Struct(), 'batches': 0}, restart_tf = True):

    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        loss_value = self.__loss()
      grad = tape.gradient(loss_value, self.u_model.trainable_variables)
      grad_flat = []
      for g in grad:
        grad_flat.append(tf.reshape(g, [-1]))
      grad_flat =  tf.concat(grad_flat, 0)
      return loss_value, grad_flat

    self.logger.log_train_start(self)

    for n in range(coupled_optimizer['batches']):

      if restart_tf or n == 0:

        self.logger.log_train_opt("Adam")
        for epoch in range(tf_epochs):
          # Optimization step
          loss_value, grads = self.__grad()
          self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
          self.logger.log_train_epoch(epoch, loss_value)
        
        self.logger.log_train_opt("LBFGS")

      print('Working on batch {0}'.format(n))

      lbfgs(loss_and_flat_grad,
        self.get_weights(),
        coupled_optimizer['nt_config'], Struct(), True,
        lambda epoch, loss, is_iter:
          self.logger.log_train_epoch(epoch, loss, "", is_iter))

      #print('Optimizing Lagrange multipliers')
      bnds = ((0., 3.), (0., 3.), (0., 3.))
      x0 = np.array([self.lambda_bc, self.lambda_u, self.lambda_v]).copy()
      res = minimize(self.return_numpy_loss, x0, method='L-BFGS-B', #jac=self.__return_jacobian, 
                     tol=1e-6, bounds=bnds, options={'maxiter': 10, 'eps': 1e-4})
      #print(res)
      [self.lambda_bc, self.lambda_u, self.lambda_v] = res.x
      #print(np.array([self.lambda_bc, self.lambda_u, self.lambda_v]))

    self.logger.log_train_end(tf_epochs + coupled_optimizer['nt_config'].maxIter)


  def f_model_predict(self, new_pred=False, X_u=[]):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime

    if new_pred:
    
      with tf.GradientTape(persistent=True) as tape:

        x_loc = tf.convert_to_tensor(X_u[...,0], dtype=self.dtype)
        y_loc = tf.convert_to_tensor(X_u[...,1], dtype=self.dtype)

        # Watching the two inputs we’ll need later, x and t
        tape.watch(x_loc)
        tape.watch(y_loc)

        # Creating prediction array
        X_u = tf.stack([x_loc, y_loc], axis=1)

        # Getting the prediction
        psi_and_p = self.u_model(X_u)

        psi = psi_and_p[..., 0, tf.newaxis]
        p = psi_and_p[..., 1, tf.newaxis]

      # Getting velocities
      u =  tape.gradient(psi, y_loc)
      v = -tape.gradient(psi, x_loc)

      # Letting the tape go
      del tape

      # Buidling the PINNs
      return u, v, p[...,0]

    else:

      with tf.GradientTape(persistent=True) as tape:

        # Watching the two inputs we’ll need later, x and t
        tape.watch(self.x_f)
        tape.watch(self.y_f)

        # Packing together the inputs
        X_f = tf.stack([self.x_f[:,0], self.y_f[:,0]], axis=1)

        # Getting the prediction
        psi_and_p = self.u_model(X_f)
        psi = psi_and_p[..., 0, tf.newaxis]
        p = psi_and_p[..., 1, tf.newaxis]

        # Getting velocities
        u = tape.gradient(psi, self.y_f)[..., 0]
        v = -tape.gradient(psi, self.x_f)[..., 0]

      # Letting the tape go
      del tape

      # Buidling the PINNs
      return u, v, p

  def predict(self, X_u):
    
    X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    u, v, p = self.f_model_predict(new_pred=True, X_u=X_u)
    
    return u, v, p

  def return_loss(self):
    return self.__loss()

  def save_model(self, name):
    self.u_model.save(name)

  def load_model(self, name):
    self.u_model = tf.keras.models.load_model(name)

  def restart_model(self):    
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))