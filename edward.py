import numpy as np
import tensorflow as tf

max_angle = np.pi / 3.
angle_spreading = 1
max_folded_length = 5
folded_length_spreading = 1
n_units = 5
n_params = 5
n_points = 10
R = 10
L_init = 3
initial_angle = np.pi / 3.

def normalize( x, min_value, max_value, spreading):
    mean_tensor = tf.constant( (max_value-min_value) / 2. , tf.float32)
    ampl = (max_value+min_value) / 2.
    x_spread = tf.scalar_mul(spreading, x)
    x_normalized = tf.erf(x_spread)
    x_normalized2 = mean_tensor + tf.scalar_mul(ampl_tensor, x_normalized)
    return x_normalized2

def init_param(n_params, n_points):
    # lambda, gamma, delta, mu, l
    params = tf.Variable( tf.random_normal([n_params-1, n_points]) )
    L = tf.Variable( tf.random_normal( [1, n_points] ) )

    return [tf.Variable(params), tf.Variable(L)]

def compute_dist(vec):

    square = tf.square( vec )
    sumsquare = tf.reduce_sum( square, axis=1 )
    return tf.sqrt(sumsquare)

def unit_output(parameters, folded_length1_, folded_length2_, origin):

    u_, v_, gamma_, delta_ = tf.slice(parameters, [1, 1, 1, 1, 1, 1], axis=1)
    u = normalize( u_, 0, 1, 1 )
    v = normalize( v_, 0, 1, 1 )
    gamma = normalize( gamma_, -max_angle, max_angle, angle_spreading )
    delta = normalize( delta_, -max_angle, max_angle, angle_spreading )
    folded_length1 = normalize( folded_length1_, 0, max_folded_length, folded_length_spreading )
    folded_length2 = normalize( folded_length2_, 0, max_folded_length, folded_length_spreading )

    a = u*folded_length1
    b = (1-u)*folded_length1
    c = v*folded_length1
    d = (1-v)*folded_length1

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
    OD_x0 = A0_x0 + sin*AD_x1 - cos*AD_y1
    OD_y0 = A0_y0 + sin*AD_x1 - cos*AD_y1

    return tf.concat( [OD_x0, OD_y0, OC_x0, OC_y0], axis=1 )

sess = tf.InteractiveSession()

coords = []
folded_length = []
parameters = []

output_initial_location = tf.constant( (R*cos(initial_angle), R*cos(initial_angle)), tf.float32 )
output_radius = tf.constant( R, tf.float32 )

B_input = tf.placeholder(tf.float32, shape=[n_points, 2])
A_input = tf.zeros(shape=[n_points, 2])


coords[0] = tf.concat( [B_input, A_input], axis=1 )
folded_length[0],

for i in range(n_units):

    parameters[i], L[i] = init_param(n_params, n_points)
    units[i+1] = unit_output(parameters[i],units[i])

loss = tf.losses.mean_squared_error( units[n_units], output_target )
