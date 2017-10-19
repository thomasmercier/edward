import numpy as np
import tensorflow as tf

def init_param(shape):

    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_dist(vec):

    square = tf.square( vec )
    sumsquare = tf.reduce_sum( square, axis=1 )
    return tf.sqrt(sumsquare)

def unit_output(parameters, origin):

    a, b, c, d, gamma, delta = tf.slice(parameters, [1, 1, 1, 1, 1, 1], axis=1)

    A0, B0 = tf.split(origin, [2, 2], axis=1)
    vecAB = B-A
    u1 = compute_dist(vecAB)

    alpha = tf.asin( (a**2 - b**2 + u1**2) / (2*a*u1) )
    beta = tf.asin( (a**2 - b**2 - u1**2) / (2*b*u1) )

    AC_x1 = a*tf.cos(alpha) + c*tf.cos(alpha+gamma)
    AC_y1 = a*tf.sin(alpha) + c*tf.sin(alpha+gamma)
    AD_x1 = a*tf.cos(alpha) + d*tf.cos(beta+delta)
    AD_y1 = a*tf.sin(alpha) + d*tf.sin(beta+delta)

    OA_x0, OA_y0 = tf.slice( A0, [1, 1], axis=1 )
    AB_x0, AB_y0 = tf.slice(vecAB, [1, 1], axis=0)
    cos = AB_x0 / u1
    sin = - AB_x0 / u1

    OC_x0 = A0_x0 + cos*AC_x1 - sin*AC_y1
    OC_y0 = A0_y0 + sin*AC_x1 - cos*AC_y1
    OD_x0 = a*tf.cos(alpha) + d*tf.cos(beta+delta)
    OD_y0 = a*tf.sin(alpha) + d*tf.sin(beta+delta)





    return origin


n_units = 5

sess = tf.InteractiveSession()

units = []
parameters = []

output_target = tf.placeholder(tf.float32, shape=[None, 2])
yB0 = tf.placeholder(tf.float32, shape=[None, 1])
xB0 = tf.zeros(shape=[None, 1])
xA0 = tf.zeros(shape=[None, 1])
yA0 = tf.zeros(shape=[None, 1])

units[0] = tf.concat( [xA0, yA0, xB0, yB0], axis=1 )

for i in range(n_units):

    parameters[i] = init_param([6])
    units[i+1] = unit_output(parameters[i],units[i])

loss = tf.losses.mean_squared_error( units[n_units], output_target )
