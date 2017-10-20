import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

max_angle = np.pi / 10.
max_folded_length = 5
n_units = 5
n_params = 5
n_points = 100
L_init = 3

def normalize( x, min_value, max_value, spreading):

    mean_tensor = tf.constant( (max_value+min_value) / 2. , tf.float32)
    ampl_tensor = tf.constant( (max_value-min_value) / 2. , tf.float32)
    spreading_tensor =  tf.reciprocal( tf.constant( spreading , tf.float32))
    x_spread = tf.scalar_mul(spreading_tensor, x)
    x_normalized = tf.erf(x_spread)
    x_normalized2 = mean_tensor + tf.scalar_mul(ampl_tensor, x_normalized)
    return x_normalized2

def init_param(n_params):

    u = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    v = tf.truncated_normal( shape=[1, 1], mean=0.5, stddev=0.25)
    gamma = tf.truncated_normal( shape=[1, 1], mean=0, stddev=max_angle/2.)
    delta = tf.truncated_normal( shape=[1, 1], mean=0, stddev=max_angle/2.)
    params = tf.concat( [u, v, gamma, delta], axis=0 )
    L = tf.truncated_normal( shape=[1, 1], mean=max_folded_length/2., stddev=max_folded_length/4.)
    return [ tf.Variable(params), tf.Variable(L) ]

def compute_dist(vec):

    square = tf.square( vec )
    sumsquare = tf.reduce_sum( square, axis=1 )
    return tf.reshape( tf.sqrt(sumsquare), [-1, 1] )

def unit_output(parameters, folded_length1, folded_length2, origin):

    u, v, gamma, delta = tf.split(parameters, [1, 1, 1, 1], axis=0)

    a = u*folded_length1
    b = (1-u)*folded_length1
    c = v*folded_length2
    d = (1-v)*folded_length2

    A0, B0 = tf.split(origin, [2, 2], axis=1)
    vecAB = B0-A0
    u1 = compute_dist(vecAB)

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

    # DEBUG ----------- sin cos OK

    OC_x0 = OA_x0 + cos*AC_x1 - sin*AC_y1
    OC_y0 = OA_y0 + sin*AC_x1 + cos*AC_y1
    OD_x0 = OA_x0 + cos*AD_x1 - sin*AD_y1
    OD_y0 = OA_y0 + sin*AD_x1 + cos*AD_y1
    OE_x0 = OA_x0 + cos*AE_x1 - sin*AE_y1
    OE_y0 = OA_y0 + sin*AE_x1 + cos*AE_y1

    temp_debug = tf.concat( [alpha, beta, AC_x1, AC_y1, AD_x1, AD_y1, sin, cos, AB_x0, AB_y0, u1], axis=1 )
    geom = tf.concat( [a, b, c, d, gamma, delta], axis=1 )
    output_coords = tf.concat( [OD_x0, OD_y0, OC_x0, OC_y0], axis=1 )
    middle_point = tf.concat( [OE_x0, OE_y0], axis=1 )

    return [temp_debug, geom, output_coords, middle_point]

sess = tf.InteractiveSession()

coords = [None] * (n_units+1)
geom = [None] * n_units
temp_debug = [None] * n_units
middle_point = [None] * n_units
folded_length = [None] * (n_units+1)
parameters = [None] * n_units

output_target = tf.placeholder( tf.float32, shape=[n_points, 2] )
B_input = tf.placeholder(tf.float32, shape=[n_points, 2])
A_input = tf.zeros(shape=[n_points, 2])


coords[0] = tf.concat( [A_input, B_input], axis=1 )
folded_length[0] = tf.constant(L_init, tf.float32, shape=[1,1])

for i in range(n_units):

    parameters[i], folded_length[i+1] = init_param(n_params)
    temp_debug[i], geom[i], coords[i+1], middle_point[i] = \
        unit_output(parameters[i], folded_length[i+1], folded_length[i], coords[i] )

output, _ = tf.split( coords[n_units], [2, 2], axis=1 )
loss = tf.losses.mean_squared_error( output_target, output )
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
tf.global_variables_initializer().run()

def test():

    A_input_ = np.array( [ [0, 0] ], dtype=np.float32 )
    B_input_ = np.array( [ [1, 1] ], dtype=np.float32 )
    for i in range(n_units):
        temp_param = tf.constant( [0, 0, 0, 0], dtype=tf.float32, shape=(4,1) )
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
        temp_param = tf.constant( [0.1, -0.2, 0, 0], dtype=tf.float32, shape=(4,1) )
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
        tf.global_variables_initializer()
        print('j=' + str(j))
        for i in range(n_train):
            loss_ = loss.eval(feed_dict=feed_dict)
            print(loss_)
            if np.isnan(loss_):
                break;
            sess.run(train_step, feed_dict=feed_dict)

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


animate_result()
