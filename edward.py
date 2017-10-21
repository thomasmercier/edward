import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

max_angle = np.pi / 6.
max_folded_length = 5
n_units = 5
n_params = 5
n_points = 100
L_max = tf.constant(5, dtype=tf.float32)

def normalize( x, min_value, max_value, spreading):

    mean_tensor = tf.constant( (max_value+min_value) / 2. , tf.float32)
    ampl_tensor = tf.constant( (max_value-min_value) / 2. , tf.float32)
    spreading_tensor =  tf.reciprocal( tf.constant( spreading , tf.float32))
    x_spread = tf.scalar_mul(spreading_tensor, x)
    x_normalized = tf.erf(x_spread)
    x_normalized2 = mean_tensor + tf.scalar_mul(ampl_tensor, x_normalized)
    return x_normalized2

def init_param():

    u = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    v = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    gamma = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    delta = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    L1 = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25 )
    L2 = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25 )
    params = tf.concat( [u, v, gamma, delta, L1, L2], axis=0 )
    return tf.Variable(params)

def compute_dist(vec):

    square = tf.square( vec )
    sumsquare = tf.reduce_sum( square, axis=1 )
    return tf.reshape( tf.sqrt(sumsquare), [-1, 1] )

def normalize(x, lb, hb):
    # 0 < x < 1
    return lb + (hb-lb)*x

def unit_output(parameters, origin):

    u_, v, gamma_, delta_, L1, L2 = tf.split(parameters, [1, 1, 1, 1, 1, 1], axis=0)
    A0, B0 = tf.split(origin, [2, 2], axis=1)
    vecAB = B0-A0

    u1 = compute_dist(vecAB)
    min_length_in = tf.reshape( tf.reduce_min(u1, axis=0), [-1, 1] )
    max_length_in = tf.reshape( tf.reduce_max(u1, axis=0), [-1, 1] )

    folded_length_in = normalize(L1, max_length_in, L_max)
    length_out = L2 * L_max

    temp = tf.minimum( min_length_in/folded_length_in, tf.ones([1, 1]) )
    u = normalize( u_, 0.5*(1-temp), 0.5*(1+temp) )
    gamma = normalize( gamma_, -max_angle, max_angle )
    delta = normalize( delta_, -max_angle, max_angle )


    a = u*folded_length_in
    b = (1-u)*folded_length_in
    c = v*length_out
    d = (1-v)*length_out

    alpha = tf.asin( (a**2 - b**2 + u1**2) / (2*a*u1) )
    beta = tf.asin( (a**2 - b**2 - u1**2) / (2*b*u1) )

    AE_x1 = a*tf.cos(alpha)
    AE_y1 = a*tf.sin(alpha)
    AC_x1 = AE_x1 + c*tf.cos(alpha+gamma)
    AC_y1 = AE_y1 + c*tf.sin(alpha+gamma)
    AD_x1 = AE_x1 + d*tf.cos(beta+delta)
    AD_y1 = AE_y1 + d*tf.sin(beta+delta)

    OA_x0, OA_y0 = tf.split( A0, [1, 1], axis=1 )
    AB_x0, AB_y0 = tf.split(vecAB, [1, 1], axis=1)
    cos = AB_y0 / u1
    sin = - AB_x0 / u1

    OC_x0 = OA_x0 + cos*AC_x1 - sin*AC_y1
    OC_y0 = OA_y0 + sin*AC_x1 + cos*AC_y1
    OD_x0 = OA_x0 + cos*AD_x1 - sin*AD_y1
    OD_y0 = OA_y0 + sin*AD_x1 + cos*AD_y1
    OE_x0 = OA_x0 + cos*AE_x1 - sin*AE_y1
    OE_y0 = OA_y0 + sin*AE_x1 + cos*AE_y1

    temp_debug = tf.concat( [alpha, beta, AC_x1, AC_y1, AD_x1, AD_y1, sin, cos, AB_x0, AB_y0], axis=1 )
    geom = tf.concat( [min_length_in, max_length_in, folded_length_in, length_out, a, b, c, d, gamma, delta], axis=1 )
    output_coords = tf.concat( [OD_x0, OD_y0, OC_x0, OC_y0], axis=1 )
    middle_point = tf.concat( [OE_x0, OE_y0], axis=1 )

    return [temp_debug, geom, output_coords, middle_point]

sess = tf.InteractiveSession()

coords = [None] * (n_units+1)
geom = [None] * n_units
temp_debug = [None] * n_units
middle_point = [None] * n_units
parameters = [None] * n_units

output_target = tf.placeholder( tf.float32, shape=[n_points, 2] )
B_input = tf.placeholder(tf.float32, shape=[n_points, 2])
A_input = tf.zeros(shape=[n_points, 2])


coords[0] = tf.concat( [A_input, B_input], axis=1 )

for i in range(n_units):

    parameters[i] = init_param()
    temp_debug[i], geom[i], coords[i+1], middle_point[i] = \
        unit_output( parameters[i], coords[i] )

output, _ = tf.split( coords[n_units], [2, 2], axis=1 )
loss = tf.losses.mean_squared_error( output_target, output )
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
tf.global_variables_initializer().run()

def test():

    A_input_ = np.array( [ [0, 0] ], dtype=np.float32 )
    B_input_ = np.array( [ [0, 1] ], dtype=np.float32 )
    for i in range(n_units):
        temp_param = tf.constant( [0.5, 0.5, 0, 0], dtype=tf.float32, shape=(4,1) )
        temp_folded_length =  tf.constant( [3], dtype=tf.float32, shape=(1,1) )
        #sess.run(tf.assign( parameters[i], temp_param ))
        #sess.run(tf.assign( folded_length[i+1], temp_folded_length ))
        feed_dict={A_input: A_input_, B_input: B_input_}
        print('------------------------1------------------------------')
        print(temp_debug[i].eval(feed_dict=feed_dict))
        print(geom[i].eval(feed_dict=feed_dict))
        print(coords[i+1].eval(feed_dict=feed_dict))
        print('------------------------2------------------------------')

def test_animate():

    # remember to set n_points = 5

    A_input_ = np.array( [ [0, 0], [0, 0], [0, 0], [0, 0], [0, 0] ], dtype=np.float32 )
    B_input_ = np.array( [ [0, 1.25], [0, 1.5], [0, 1.75], [0, 2], [0, 2.25] ], dtype=np.float32 )
    for i in range(n_units):
        temp_param = tf.constant( [0.5, 0.5, 0, 0], dtype=tf.float32, shape=(4,1) )
        temp_folded_length =  tf.constant( [3], dtype=tf.float32, shape=(1,1) )
        sess.run(tf.assign( parameters[i], temp_param ))
        sess.run(tf.assign( folded_length[i+1], temp_folded_length ))
    feed_dict={A_input: A_input_, B_input: B_input_}
    coords_ = [None]*(n_points+1)
    for i in range(n_points+1):
        coords_[i] = coords[i].eval(feed_dict=feed_dict)
    middle_point_ = [None]*n_points
    for i in range(n_points):
        middle_point_[i] = middle_point[i].eval(feed_dict=feed_dict)

    fig = plt.figure()
    ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))
    lines = [None]*n_points*4
    for i in range(n_points):
        lines[4*i+0], = ax.plot([], [], 'bo-')
        lines[4*i+1], = ax.plot([], [], 'ro-')
        lines[4*i+2], = ax.plot([], [], 'bo-')
        lines[4*i+3], = ax.plot([], [], 'ro-')

    def init():
        for i in range(n_points):
            for j in range(4):
                lines[4*i+j].set_data([], [])
        return tuple(lines)

    def animate(i):
        for j in range(n_units):
            xA, yA, xB, yB = coords_[j][i,:]
            xD, yD, xC, yC = coords_[j+1][i,:]
            xE, yE = middle_point_[j][i,:]
            lines[4*j+0].set_data( [xA, xE], [yA, yE] )
            lines[4*j+1].set_data( [xB, xE], [yB, yE] )
            lines[4*j+2].set_data( [xC, xE], [yC, yE] )
            lines[4*j+3].set_data( [xD, xE], [yD, yE] )
        return tuple(lines)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_points, interval=200, blit=True)
    plt.show()

def animate_result():

    # remember to set n_points = 100

    R = 5
    center = [0, -5]
    theta_min = np.pi / 3.
    theta_max = np.pi / 6.
    y_min = 1
    y_max = 1.5
    n_train = 1000

    A_input_ = np.zeros( (n_points, 2) )
    B_input_ = np.empty( (n_points, 2) )
    output_target_ =  np.empty( (n_points, 2) )

    for i in range(n_points):
        y = y_min + (y_max-y_min) * i / float(n_points-1)
        theta = theta_min + (theta_max-theta_min) * i / float(n_points-1)
        B_input_[i,0] = 0
        B_input_[i,1] = y
        output_target_[i,0] = center[0] + R*np.cos(theta)
        output_target_[i,1] = center[1] + R*np.sin(theta)

    feed_dict={A_input: A_input_, B_input: B_input_, output_target:output_target_}

    for j in range(n_train):
        print(j)
        loss_ = loss.eval(feed_dict=feed_dict)
        print(loss_)
        sess.run(train_step, feed_dict=feed_dict)

    coords_ = [None]*(n_units+1)
    for i in range(n_units+1):
        coords_[i] = coords[i].eval(feed_dict=feed_dict)
    middle_point_ = [None]*n_units
    for i in range(n_units):
        middle_point_[i] = middle_point[i].eval(feed_dict=feed_dict)

    fig = plt.figure()
    ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))
    lines = [None]*n_points*4
    for i in range(n_points):
        lines[4*i+0], = ax.plot([], [], 'bo-')
        lines[4*i+1], = ax.plot([], [], 'ro-')
        lines[4*i+2], = ax.plot([], [], 'bo-')
        lines[4*i+3], = ax.plot([], [], 'ro-')

    def init():
        for i in range(n_points):
            for j in range(4):
                lines[4*i+j].set_data([], [])
        return tuple(lines)

    def animate(i):
        for j in range(n_units):
            xA, yA, xB, yB = coords_[j][i,:]
            xD, yD, xC, yC = coords_[j+1][i,:]
            xE, yE = middle_point_[j][i,:]
            lines[4*j+0].set_data( [xA, xE], [yA, yE] )
            lines[4*j+1].set_data( [xB, xE], [yB, yE] )
            lines[4*j+2].set_data( [xC, xE], [yC, yE] )
            lines[4*j+3].set_data( [xD, xE], [yD, yE] )
        return tuple(lines)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_points, interval=20, blit=True)
    plt.show()


animate_result()
