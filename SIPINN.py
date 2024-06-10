"""""
M1: PINN
M2: IAPINN(PINN with improved architecture)
M3: SIPINN
"""""


import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs


# 初始化每层变量
# Initialize variables for each layer
def xavier_init(size,i):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
    return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                       dtype=tf.float32, name='w'+str(i))

# 初始化网络层
# Initialize network layer
def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = xavier_init(size=[layers[l], layers[l + 1]], i = l)
        # Bias b is initialized to 0     偏置b被初始化为0
        b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32, name='b'+str(l))
        weights.append(W)
        biases.append(b)
    return weights, biases

# 前向传播
# Forward propagation
def forward_pass(H):
    if model == 'M1':
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b, name='M1')
        return H

    if model in ['M2', 'M3']:
        num_layers = len(layers)
        encoder_1 = tf.tanh(tf.add(tf.matmul(H, encoder_weights_1), encoder_biases_1))
        encoder_2 = tf.tanh(tf.add(tf.matmul(H, encoder_weights_2), encoder_biases_2))
    
        if model == 'M2':
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                H = tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + \
                    tf.math.multiply(tf.tanh(tf.add(one - tf.matmul(H, W), b)), encoder_2)
            W = weights[-1]
            b = biases[-1]
            H = tf.add(tf.matmul(H, W), b, name='M2')
    
        if model == 'M3':
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                H = tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + encoder_2
        
            W = weights[-1]
            b = biases[-1]
            H = tf.add(tf.matmul(H, W), b, name='M3')
        return H

# 分割网络输出
# Split the output of the network
def sep(target):
    a1 = target[:, 0:1]
    a2 = target[:, 1:2]
    a3 = target[:, 2:3]
    a4 = target[:, 3:4]
    return a1, a2, a3, a4


def sample(N1, N2):
    """
    归一化样本点，以保证网络的输入量纲一致。
    即:  r: [0, 0.3] --> [0, 1]    z: [0, 0.18] --> [0, 1]
    由于物理量量纲的转变会对方程产生影响，因此重新计算了空间坐标的量纲转化后的方程
    :param N1: Number of sampling points per boundary
    :param N2: Number of collocation points
    :return: 
    """

    # Sample Range  采样范围
    rlb = 0
    rub = 1
    zlb = 0
    zub = 1

    wx = n2 * 2 * 3.1416 / 60     # 0.00031416
    wc = -n1 * 2 * 3.1416 / 60    # -0.00031416

    r = np.linspace(rlb, rub, num=N1, dtype=None).reshape(N1, 1)
    z = np.linspace(zlb, zub, num=N1, dtype=None).reshape(N1, 1)
    print("The Initial condition of the furnace body have been completely set!")


    # collocation points  配置点
    r_col = lhs(1, N2)
    z_col = lhs(1, N2)


    # Boundary points   边界点
    # ①Axis of symmetry
    b1_r = np.zeros((N1, 1))
    b1_z = z
    b1_vr = np.zeros((N1, 1))
    b1_vT = np.zeros((N1, 1))
    
    # ②Crucible bottom
    b2_r = r
    b2_z = np.zeros((N1, 1))
    b2_vr = np.zeros((N1, 1))
    b2_vz = np.zeros((N1, 1))
    b2_vT = b2_r * wc
    
    # ③Crucible wall
    b3_r = np.ones((N1, 1)) * rub
    b3_z = z
    b3_vr = np.zeros((N1, 1))
    b3_vz = np.zeros((N1, 1))
    b3_vT = b3_r * wc
    
    # ④Interface
    b4_r = np.linspace(0, 0.5, num=N1, dtype=None).reshape(N1, 1)
    b4_z = np.ones((N1, 1)) * zub
    b4_vr = np.zeros((N1, 1))
    b4_vz = np.zeros((N1, 1))
    b4_vT = b4_r * wx
    
    # ⑤Free surface
    b5_r = np.linspace(0.5 + 0.5 / N1, 1, num=N1, dtype=None).reshape(N1, 1)
    b5_z = np.ones((N1, 1)) * zub
    b5_vr = np.zeros((N1, 1))
    b5_vz = np.zeros((N1, 1))
    b5_vT = np.linspace(b4_vT[-1, 0], b3_vT[0, 0], num=N1, dtype=None).reshape(N1, 1)

    return r_col, z_col, \
           b1_r, b1_z, b1_vr / vs, b1_vT / vs, \
           b2_r, b2_z, b2_vr / vs, b2_vz / vs, b2_vT / vs, \
           b3_r, b3_z, b3_vr / vs, b3_vz / vs, b3_vT / vs, \
           b4_r, b4_z, b4_vr / vs, b4_vz / vs, b4_vT / vs, \
           b5_r, b5_z, b5_vr / vs, b5_vz / vs, b5_vT / vs


# 计算边界损失
# Calculate loss of boundary points
def bound_loss(b1_r, b1_z, b1_vr, b1_vT,
               b2_r, b2_z, b2_vr, b2_vz, b2_vT,
               b3_r, b3_z, b3_vr, b3_vz, b3_vT,
               b4_r, b4_z, b4_vr, b4_vz, b4_vT,
               b5_r, b5_z, b5_vr, b5_vz, b5_vT,
               model):

    p1 = model(tf.concat([b1_r, b1_z], 1))
    p2 = model(tf.concat([b2_r, b2_z], 1))
    p3 = model(tf.concat([b3_r, b3_z], 1))
    p4 = model(tf.concat([b4_r, b4_z], 1))
    p5 = model(tf.concat([b5_r, b5_z], 1))

    p1_vr, p1_vz, p1_vT, _ = sep(p1)
    p2_vr, p2_vz, p2_vT, _ = sep(p2)
    p3_vr, p3_vz, p3_vT, _ = sep(p3)
    p4_vr, p4_vz, p4_vT, _ = sep(p4)
    p5_vr, p5_vz, p5_vT, _ = sep(p5)


    # ①Axis of symmetry
    p1_vz_r = tf.gradients(p1_vz, b1_r)
    b1_loss = tf.add_n([tf.reduce_mean(tf.square(b1_vr - p1_vr)),
                        tf.reduce_mean(tf.square(b1_vT - p1_vT)),
                        tf.reduce_mean(tf.square(p1_vz_r))], name='b1_loss')
    
    # ②Crucible bottom
    b2_loss = tf.add_n([tf.reduce_mean(tf.square(b2_vr - p2_vr)),
                        tf.reduce_mean(tf.square(b2_vz - p2_vz)),
                        tf.reduce_mean(tf.square(b2_vT - p2_vT))], name='b2_loss')
    # ③Crucible wall
    b3_loss = tf.add_n([tf.reduce_mean(tf.square(b3_vr - p3_vr)),
                        tf.reduce_mean(tf.square(b3_vz - p3_vz)),
                        tf.reduce_mean(tf.square(b3_vT - p3_vT))], name='b3_loss')
    # ④Interface
    b4_loss = tf.add_n([tf.reduce_mean(tf.square(b4_vr - p4_vr)),
                        tf.reduce_mean(tf.square(b4_vz - p4_vz)),
                        tf.reduce_mean(tf.square(b4_vT - p4_vT))], name='b4_loss')
    # ⑤Free surface
    b5_loss = tf.add_n([tf.reduce_mean(tf.square(p5_vr - b5_vr)),
                        tf.reduce_mean(tf.square(p5_vz - b5_vz)),
                        tf.reduce_mean(tf.square(p5_vT - b5_vT))], name='b5_loss')
    
    # Boundary loss
    mse_b = tf.add_n([b1_loss, b2_loss, b3_loss, b4_loss, b5_loss])
    return mse_b


# 计算配置点损失
# Calculate loss of collocation points
def coll_loss(r, z, rs, zs, vs, model):
    P = model(tf.concat([r, z], 1))
    vr, vz, vT, P = sep(P)

    # 恢复速度量纲
    # Dimension of restoration speed
    vr = vr * vs
    vz = vz * vs
    vT = vT * vs

    # eq1: vr_r, vr_z, P_r, vr_r2, vr_z2
    vr_r = tf.gradients(vr, r)
    vr_z = tf.gradients(vr, z)
    P_r = tf.gradients(P, r)
    vr_r2 = tf.gradients(vr_r, r)
    vr_z2 = tf.gradients(vr_z, z)

    # eq2: vT_r, vT_z, vT_r2, vT_z2
    vT_r = tf.gradients(vT, r)
    vT_z = tf.gradients(vT, z)
    vT_r2 = tf.gradients(vT_r, r)
    vT_z2 = tf.gradients(vT_z, z)

    # eq3:  vz_r, vz_z, P_z, vz_r2, vz_z2
    vz_r = tf.gradients(vz, r)
    vz_z = tf.gradients(vz, z)
    P_z = tf.gradients(P, z)
    vz_r2 = tf.gradients(vz_r, r)
    vz_z2 = tf.gradients(vz_z, z)

    # eqm:  vr_r、vz_z  (It's already been calculated above)
    # vr_r = tf.gradients(vr, r)
    # vz_z = tf.gradients(vz, z)


    eq1 = vr * vr_r[0] / rs - pow(vT, 2) / r / rs + vz * vr_z[0] / zs + \
          P_r[0] / rs - 0.0008 / 2530 * (
    vr_r2[0] / (rs ** 2) + vr_r[0] / r / (rs ** 2) - vr / pow(r, 2) / (rs ** 2) + vr_z2[0] / (zs ** 2))
    eq2 = vr * vT_r[0] / rs + vr * vT / r / rs + vz * vT_z[0] / zs - \
          0.0008 / 2530 * (vT_r2[0] / (rs ** 2) + vT_r[0] / r / (rs ** 2) - vT / pow(r, 2) / (rs ** 2) + vT_z2[0] / (zs ** 2))
    eq3 = vr * vz_r[0] / rs + vz * vz_z[0] / zs + \
          P_z[0] / zs + 9.81 - 0.0008 / 2530 * (
    vz_r2[0] / (rs ** 2) + vz_r[0] / r / (rs ** 2) + vz_z2[0] / (zs ** 2))

    eq1_m = tf.reduce_mean(tf.square(eq1), name="eq1")
    eq2_m = tf.reduce_mean(tf.square(eq2), name="eq2")
    eq3_m = tf.reduce_mean(tf.square(eq3), name="eq3")

    # eqm
    eqm = vr_r[0] / rs + vr / r / rs + vz_z[0] / zs
    eqm_m = tf.multiply(tf.reduce_mean(tf.square(eqm)), name='eqm')

    return eq1_m, eq2_m, eq3_m, eqm_m




# Trains the model by minimizing the MSE loss
def train(nIter, name, version, rotation_condition):

    N1 = 401   # number of boundary points     每个边界的采样点数量
    N2 = 6000  # number of collocation points  配置点的采样点数量
    
    # Sampling
    r_col, z_col, \
    b1_r, b1_z, b1_vr, b1_vT, \
    b2_r, b2_z, b2_vr, b2_vz, b2_vT, \
    b3_r, b3_z, b3_vr, b3_vz, b3_vT, \
    b4_r, b4_z, b4_vr, b4_vz, b4_vT, \
    b5_r, b5_z, b5_vr, b5_vz, b5_vT = sample(N1, N2)
    
    start_time = time.time()
    start_time2 = time.time()
    
    tf_dict = {b1_r_tf: b1_r, b1_z_tf: b1_z, b1_vr_tf: b1_vr, b1_vT_tf: b1_vT,
               b2_r_tf: b2_r, b2_z_tf: b2_z, b2_vr_tf: b2_vr, b2_vz_tf: b2_vz, b2_vT_tf: b2_vT,
               b3_r_tf: b3_r, b3_z_tf: b3_z, b3_vr_tf: b3_vr, b3_vz_tf: b3_vz, b3_vT_tf: b3_vT,
               b4_r_tf: b4_r, b4_z_tf: b4_z, b4_vr_tf: b4_vr, b4_vz_tf: b4_vz, b4_vT_tf: b4_vT,
               b5_r_tf: b5_r, b5_z_tf: b5_z, b5_vr_tf: b5_vr, b5_vz_tf: b5_vz, b5_vT_tf: b5_vT,
               r_col_tf: r_col, z_col_tf: z_col, wb_tf: wb
               }

    b_loss = sess.run(b_loss_tf, tf_dict)
    eq1, eq2, eq3, eqm = sess.run([eq1_tf, eq2_tf, eq3_tf, eqm_tf], tf_dict)
    coll_loss = sess.run(coll_loss_tf, tf_dict)
    loss_value = b_loss + coll_loss

    print("Initial error: ")
    print('Loss: %.8f, b_loss: %.8f, coll_loss: %.8f' %
          (loss_value, b_loss, coll_loss))
    print('eq1: %.8f, eq2: %.8f, eq3: %.8f, eqm: %.8f' %
          (eq1, eq2, eq3, eqm))
    
    # Adam
    for it in range(nIter):
        tf_dict = {b1_r_tf: b1_r, b1_z_tf: b1_z, b1_vr_tf: b1_vr, b1_vT_tf: b1_vT,
                   b2_r_tf: b2_r, b2_z_tf: b2_z, b2_vr_tf: b2_vr, b2_vz_tf: b2_vz, b2_vT_tf: b2_vT,
                   b3_r_tf: b3_r, b3_z_tf: b3_z, b3_vr_tf: b3_vr, b3_vz_tf: b3_vz, b3_vT_tf: b3_vT,
                   b4_r_tf: b4_r, b4_z_tf: b4_z, b4_vr_tf: b4_vr, b4_vz_tf: b4_vz, b4_vT_tf: b4_vT,
                   b5_r_tf: b5_r, b5_z_tf: b5_z, b5_vr_tf: b5_vr, b5_vz_tf: b5_vz, b5_vT_tf: b5_vT,
                   r_col_tf: r_col, z_col_tf: z_col, wb_tf: wb
                   }
        
        sess.run(optimizer1_op, tf_dict)
        
        # Print
        if it % 10 == 0:
            elapsed = time.time() - start_time

            b_loss = sess.run(b_loss_tf, tf_dict)
            eq1, eq2, eq3, eqm = sess.run([eq1_tf, eq2_tf, eq3_tf, eqm_tf], tf_dict)
            coll_loss = sess.run(coll_loss_tf, tf_dict)
            loss_value = b_loss + coll_loss

            loss_log.append(loss_value)
            
            print("==========================================================================")
            print('It: %d, Loss: %.8f, b_loss: %.8f, coll_loss: %.8f, Time: %.2f' %
                  (it, loss_value, b_loss, coll_loss, elapsed))
            print('eq1: %.8f, eq2: %.8f, eq3: %.8f, eqm: %.8f' %
                  (eq1, eq2, eq3, eqm))
            start_time = time.time()

    
    
    # 全部训练完成，最后打印一次误差
    # Training completed, last print error
    b_loss = sess.run(b_loss_tf, tf_dict)
    eq1, eq2, eq3, eqm = sess.run([eq1_tf, eq2_tf, eq3_tf, eqm_tf], tf_dict)
    coll_loss = sess.run(coll_loss_tf, tf_dict)
    loss_value = b_loss + coll_loss
    
    loss_log.append(loss_value)
    
    print("==========================================================================")
    print("Final error: ")
    print('It: %d, Loss: %.8f, b_loss: %.8f, coll_loss: %.8f, Time: %.2f' %
          (nIter, loss_value, b_loss, coll_loss, elapsed))
    print('eq1: %.8f, eq2: %.8f, eq3: %.8f, eqm: %.8f' %
          (eq1, eq2, eq3, eqm))
    
    print("Total training duration: ", (time.time() - start_time2) / 3600, " h")
    
    # Save the model   保存模型
    saver.save(sess, name + "/" + name)
    # Save loss value  保存损失
    name = 'loss_log/net' + str(version)
    os.makedirs(name)
    np.savetxt(name + "/loss.txt", loss_log, fmt='%0.8f')


def draw(target, title):
    min = np.min(target.flatten())
    max = np.max(target.flatten())

    fig, ax = plt.subplots()
    h = ax.imshow(target, interpolation='nearest', cmap='rainbow', vmin=min, vmax=max,
                  extent=[0, 0.3, 0, 0.18],
                  origin='lower', aspect='auto')
    # 插值方式：nearest；颜色类型：rainbow
    # interpolation method：nearest；Color Type：rainbow
    plt.title(title, pad=15, loc='center', fontdict={'family': 'Times New Roman', 'fontsize': 20})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    plt.show()


def picture():
    rlb = 0
    rub = 1
    zlb = 0
    zub = 1

    N = 101

    # Sampling
    r = np.linspace(rlb, rub, num=N, dtype=None).reshape(N, 1)  # [0, 0.003, 0.006, .... ,0.3]
    z = np.linspace(zlb, zub, num=N, dtype=None).reshape(N, 1)
    r = r[1:-1]
    z = z[1:-1]
    A, B = np.meshgrid(r, z)
    X_star = np.hstack((A.flatten()[:, None], B.flatten()[:, None]))
    r_col = X_star[:, 0].reshape((N - 2) ** 2, 1)
    z_col = X_star[:, 1].reshape((N - 2) ** 2, 1)

    tf_dict = {r_col_tf: r_col, z_col_tf: z_col}
    Pred = sess.run(p_coll, tf_dict)
    vr, vz, vT, P = sep(Pred)


    V = np.zeros(N * N, )
    for i in range(N * N):
        V[i] = (vr[i] ** 2 + vz[i] ** 2 + vT[i] ** 2) ** (1 / 2)

    V = V.reshape(N, N) * vs
    P = P.reshape(N, N)

    P = P.flatten()
    P_min = P.min()
    P = (P - P_min) * 2530
    P = P.reshape(N, N)

    draw(V, "V")
    draw(P, "P")



"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
print("START！")
start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""
在训练模型时，只需要调整以下三个参数：
 - version：模型序号
 - rotation_condition：旋转条件
 - model：选择的网络架构
 
When training the model, only the following three parameters need to be adjusted:
 - Version: The serial number of the model
 - Rotation_ Condition: Rotation condition
 - Model: Selected network architecture
"""
version = 1
rotation_condition = 1
model = 'M1'
""""""

layers = [2, 128, 128, 128, 128, 128, 4]
lnn = 0.002   # Learning rate
lnde = 0.9    # Attenuation rate of Learning rate


# n1: Rotation rate of crucible  坩埚转速
# n2: Rotation rate of crystal   晶体转速
if rotation_condition == 1:    # 晶体坩埚同时旋转
    n1 = 0.003
    n2 = 0.003
elif rotation_condition == 2:  # 仅坩埚
    n1 = 0.003
    n2 = 0
elif rotation_condition == 3:  # 仅晶体
    n1 = 0
    n2 = 0.003


# Set parameters
Config = tf.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))


# Boundary loss weight  边界损失的权重
wb = np.array(2000.0)

# 初始化网络结构与隐层变量
# Initialize network structure and hidden layer variables
weights, biases = initialize_NN(layers)
if model in ['M2', 'M3']:
    encoder_weights_1 = xavier_init([2, layers[1]], 'ew1')
    encoder_biases_1 = xavier_init([1, layers[1]], 'eb1')
    
    encoder_weights_2 = xavier_init([2, layers[1]], 'ew2')
    encoder_biases_2 = xavier_init([1, layers[1]], 'eb2')


# Normalization   归一化
rs = 0.3
zs = 0.18
vs = 0.00031416 * 0.3

one = tf.constant(1, dtype=tf.float32, name="one")


# Definition graph   定义图
b1_r_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b1_r')
b1_z_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b1_z')
b1_vr_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b1_vr')
b1_vT_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b1_vT')

b2_r_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b2_r')
b2_z_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b2_z')
b2_vr_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b2_vr')
b2_vz_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b2_vz')
b2_vT_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b2_vT')

b3_r_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b3_r')
b3_z_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b3_z')
b3_vr_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b3_vr')
b3_vz_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b3_vz')
b3_vT_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b3_vT')

b4_r_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b4_r')
b4_z_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b4_z')
b4_vr_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b4_vr')
b4_vz_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b4_vz')
b4_vT_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b4_vT')

b5_r_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b5_r')
b5_z_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b5_z')
b5_vr_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b5_vr')
b5_vz_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b5_vz')
b5_vT_tf = tf.placeholder(tf.float32, shape=(None, 1), name='b5_vT')

r_col_tf = tf.placeholder(tf.float32, shape=(None, 1), name='r_col')
z_col_tf = tf.placeholder(tf.float32, shape=(None, 1), name='z_col')


# Draw
r_picture_tf = tf.placeholder(tf.float32, shape=(None, 1), name='r_picture')
z_picture_tf = tf.placeholder(tf.float32, shape=(None, 1), name='z_picture')

wb_tf = tf.placeholder(tf.float32, shape=wb.shape, name='wb_tf')


p_coll = forward_pass(tf.concat([r_col_tf, z_col_tf], 1))
p1  = forward_pass(tf.concat([b1_r_tf, b1_z_tf], 1))
p2 = forward_pass(tf.concat([b2_r_tf, b2_z_tf], 1))
p3 = forward_pass(tf.concat([b3_r_tf, b3_z_tf], 1))
p4 = forward_pass(tf.concat([b4_r_tf, b4_z_tf], 1))
p5 = forward_pass(tf.concat([b5_r_tf, b5_z_tf], 1))

eq1_tf, eq2_tf, eq3_tf, eqm_tf = coll_loss(r_col_tf, z_col_tf, rs, zs, vs, forward_pass)
coll_loss_tf = tf.add_n([eq1_tf, eq2_tf, eq3_tf, eqm_tf], name='coll_loss')

b_loss_tf = tf.multiply(wb_tf, bound_loss(b1_r_tf, b1_z_tf, b1_vr_tf, b1_vT_tf,
                                       b2_r_tf, b2_z_tf, b2_vr_tf, b2_vz_tf, b2_vT_tf,
                                       b3_r_tf, b3_z_tf, b3_vr_tf, b3_vz_tf, b3_vT_tf,
                                       b4_r_tf, b4_z_tf, b4_vr_tf, b4_vz_tf, b4_vT_tf,
                                       b5_r_tf, b5_z_tf, b5_vr_tf, b5_vz_tf, b5_vT_tf,
                                       forward_pass), name="b_loss")
loss_tf = tf.add_n([coll_loss_tf, b_loss_tf], name='loss')


# Attenuate Learning rate   衰减学习率
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = lnn
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                1000, lnde, staircase=True)

# Adam Optimizer
beta_1 = 0.99
optimizer1_op = tf.train.AdamOptimizer(learning_rate, beta_1).minimize(loss_tf,global_step=global_step)

# draw designs
p = forward_pass(tf.concat([r_picture_tf, z_picture_tf], 1))

# Record loss value  记录损失值
loss_log = []


# Saving and Loading Models   保存和加载模型
saver = tf.train.Saver()

# 初始化Tensorflow变量
# Initialize Tensorflow variable
init = tf.global_variables_initializer()
sess.run(init)
print("Time spent on graph: ", (time.time() - start_time) / 60, "minute")


# 模型保存的名称
# Name of the saved model
PINN_name = "PINN" + str(version)
train(nIter=30000, name = PINN_name, version=version, rotation_condition=rotation_condition)
picture()




