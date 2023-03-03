import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  H1 = int (1 + (x.shape[2] + 2 * pad - w.shape[2] ) / stride)
  W1 = int (1 + (x.shape[3] + 2 * pad - w.shape[3] ) / stride)
  
  if pad != 0:
    padded_x = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
  else:
    padded_x = x.copy()
        
  out = np.zeros((x.shape[0], w.shape[0], H1, W1))
  for n in range(x.shape[0]):
      for f in range(w.shape[0]):
          for i in range(H1):
              for j in range(W1):
                  out[n, f, i, j] = np.sum( padded_x[n, :, i*stride:i*stride+w.shape[2], j*stride : j*stride + w.shape[3]] * w[f]) + b[f]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #  
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _, _, H_out, W_out = dout.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  
  for n in range(N):
      for f in range(F):
          db[f] += np.sum(dout[n, f])
          for h_out in range(H_out):
              for w_out in range(W_out):
                  dw[f] += x_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] * \
                  dout[n, f, h_out, w_out]
                  dx_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] += w[f] * \
                  dout[n, f, h_out, w_out]

  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = int(1 + (H - pool_height) / stride)
  W_out = int(1 + (W - pool_width) / stride)

  out = np.zeros((N, C, H_out, W_out))

  for n in range(N):
      for c in range(C):
          for hi in range(H_out):
              for wi in range(W_out):
                  out[n, c, hi, wi] = np.max(x[n, c, hi * stride : hi * stride +
                                               pool_height, wi * stride : wi * stride
                                               + pool_width ]) 

  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H_out, W_out = dout.shape
  dx = np.zeros_like(x)


  for n in range(N):
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                # get the index in the region i,j where the value is the maximum
                i_t, j_t = np.where(np.max(x[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width])
                                    == x[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width])
                i_t, j_t = i_t[0], j_t[0]
                
                dx[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width][i_t, j_t] = dout[n, c, i, j]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
    # x is size (m,w,h,c)
    # Gamma is size of (C,)
    # Beta is size of (C.)
  N, C, H, W = x.shape 
  N = (N*H*W)
    
  mu = (1/N) * np.sum(x,axis=(0,2,3),keepdims = True)
  var = np.var(x,axis=(0,2,3),keepdims=True)
    
  normalization_denom = np.sqrt( var + eps)  
  xn = (x - mu)/normalization_denom

  out = gamma.reshape(1,C,1,1) * xn + beta.reshape(1,C,1,1)

  running_mean = bn_param.get('running_mean', np.zeros(C))
  running_var = bn_param.get('running_var', np.zeros(C))

  cache = [x, eps, mu, var, xn, normalization_denom, running_mean, running_var, gamma]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ # 
  x, eps, mu, var, xn, normalization_denom, running_mean, running_var, gamma = cache
  N, C, H, W = dout.shape
  N = (N*H*W)
  
  gamma = gamma.reshape(1,C,1,1)
    
  dbeta = np.sum(dout, axis = (0,2,3))
  dgamma = np.sum(dout * xn, axis = (0,2,3))
    
  dxn = dout * gamma
   
  normalization_denom = 1 / np.sqrt( var + eps)
  power_denom = 1/pow((var - eps),1.5)
    
  xc = x - mu
    
  dvar = (-0.5) * np.sum(power_denom * xc * dxn, axis =(0,2,3)).reshape(1,C,1,1)
  dmu = -1 * normalization_denom * np.sum(dxn, axis =(0,2,3)).reshape(1,C,1,1) - (2/N) * np.sum(xc, axis = (0,2,3)).reshape(1,C,1,1) * dvar
    
  dx = normalization_denom * dxn + (1/N) * (dmu) + (2/N) * xc * dvar
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta