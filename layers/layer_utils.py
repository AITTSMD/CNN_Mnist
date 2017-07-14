#coding:utf-8
from layers import *
from fast_layers import *

#fc+relu f
def affine_relu_forward(x, w, b):
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache,relu_cache)
  return out, cache
#fc+bn+relu f
def affine_bn_relu_forward(x, w, b,gamma,beta,bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(a,gamma,beta,bn_param)
    out,relu_cache = relu_forward(bn_out)
    cache = (fc_cache,relu_cache,bn_cache)
    return out,cache
#fc+bn f
def affine_bn_forward(x, w, b,gamma,beta,bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(a,gamma,beta,bn_param)
    cache = (fc_cache,bn_cache)
    return bn_out,cache
#fc+relu b
def affine_relu_backward(dout, cache):
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db,da
#fc+bn+relu b
def affine_bn_relu_backward(dout, cache):
    fc_cache, relu_cache,bn_cache = cache
    #以此传递残差 更新权值和偏置
    da = relu_backward(dout, relu_cache)
    dx_bn, dgamma, dbeta = batchnorm_backward(da,bn_cache)
    dx,dw,db = affine_backward(dx_bn,fc_cache)
    return dx,dw,db,da,dgamma,dbeta
#fc+bn b
def affine_bn_backward(dout, cache):
    fc_cache,bn_cache = cache
    dx_bn, dgamma, dbeta = batchnorm_backward(dout,bn_cache)
    dx,dw,db = affine_backward(dx_bn,fc_cache)
    return dx,dw,db,dgamma,dbeta
#conv+relu f
def conv_relu_forward(x, w, b, conv_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)#！！！！
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache

#conv+relu b
def conv_relu_backward(dout, cache):
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

#conv+relu+pool f
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  #conv
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  #relu
  s, relu_cache = relu_forward(a)
  #pool
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache
#conv+bn+relu f
def conv_bn_relu_forward(x, w, b,gamma,beta ,conv_param,bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  sbn_out, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(sbn_out)
  cache = (conv_cache, relu_cache ,sbn_cache)
  return relu_out,cache
#conv+bn+relu+conv+bn+relu+pool f
def conv_bn_relu_double_pool_forward(x, w, b,gamma,beta ,conv_param, pool_param,bn_param):
  cache = []
  input_data = x
  for i in range(2):
    a,conv_cache = conv_bn_relu_forward(input_data, w, b, gamma, beta, conv_param, bn_param)
    cache.append(conv_cache)
    sbn_out, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    cache.append(sbn_cache)
    relu_out, relu_cache = relu_forward(sbn_out)
    cache.append(relu_cache)
    input_data = relu_out
  pool_out, pool_cache= max_pool_forward_fast(relu_out, pool_param)
  cache.append(pool_cache)
  return pool_out,cache
    
#conv+bn+relu+pool f
def conv_bn_relu_pool_forward(x, w, b,gamma,beta ,conv_param, pool_param,bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  sbn_out, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param) 
  relu_out, relu_cache = relu_forward(sbn_out)
  pool_out, pool_cache = max_pool_forward_fast(relu_out,pool_param)
  cache = (conv_cache, relu_cache, pool_cache,sbn_cache)
  return pool_out,cache
#conv+bn+relu b
def conv_bn_relu_backward(dout, cache):
  conv_cache, relu_cache ,sbn_cache = cache
  da = relu_backward(dout, relu_cache)
  dsbn,dgamma,dbeta = spatial_batchnorm_backward(da,sbn_cache)
  dx, dw, db = conv_backward_fast(dsbn,conv_cache)
  return dx, dw, db,dgamma,dbeta
#conv+relu+pool b    
def conv_relu_pool_backward(dout, cache):
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db
#conv+bn+relu+pool b
def conv_bn_relu_pool_backward(dout, cache):
  conv_cache, relu_cache, pool_cache,sbn_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dsbn,dgamma,dbeta = spatial_batchnorm_backward(da,sbn_cache)
  dx, dw, db = conv_backward_fast(dsbn,conv_cache)
  return dx, dw, db,dgamma,dbeta
#conv+bn+relu+conv+bn+relu+pool b
def conv_bn_relu_double_pool_backward(dout,cache):
  dparams = []
  pool_cache = cache[-1]
  ds = max_pool_backward_fast(dout, pool_cache)  
  for i in range(2): 
    relu_cache = cache[5-i*3]
    da = relu_backward(ds, relu_cache)
    sbn_cache = cache[4-i*3]
    dsbn,dgamma,dbeta = spatial_batchnorm_backward(da,sbn_cache)
    conv_cache = cache[3-i*3]
    dx, dw, db = conv_backward_fast(dsbn,conv_cache)
    for item in (dw,db,dgamma,dbeta):
      dparams.append(item)
    ds = dx
  return dx,dparams 
  
    
    