# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from drl.utils.logx import log, Logger, EpochLogger, ExitClass
from drl.utils.run_utils import setup_logger_kwargs
from drl.envs.vfh_planner import vfh_plan, forward_move, get_close_vw

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.sum = np.zeros()

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def get_data_pointer(self):
        return self.data_pointer

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.end_point = [-1] #存储每个episode最后一个step对应的index
        self.is_full = False

    # def store(self, transition):
    def store(self, transition, n_features, state_dim, is_terminal):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        # p = 0
        # reward = transition[n_features+state_dim + 1]
        # if reward < -400:
        #     reward_p = abs(reward)
        # elif reward > 0:
        #     reward_p = reward +1
        # else:
        #     reward_p = reward
        # max_p = max(p, reward_p)
        # print('reward is %f'%reward)
        # print ('P is %f'%max_p)
        if is_terminal:
            index = self.tree.get_data_pointer()
            self.end_point.append(index)
            #判断memory是否存满
            if self.end_point[-1] < self.end_point[-2]:
                self.is_full = True
            #当memory存满时，删除被覆盖的episode所对应的index
            if self.is_full:
                while self.end_point[0] <= index:
                    del self.end_point[0]
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)  # 0<beta<1
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            image_size=[64, 64, 3],
            state_dim = 5,
            learning_rate=0.005,
            reward_decay=0.97,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            restore_model=False,
            exp_name = None,
            net_type='resnet',
            is_train = True,
            n_steps = 1
    ):

        logger_kwargs = setup_logger_kwargs(exp_name)
        self.logger = EpochLogger(**logger_kwargs)
        self.net_type = net_type
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.restore_model = restore_model
        self.n_features = self.image_size[0] * self.image_size[1] * self.image_size[2]
        self.state_dim = state_dim
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.is_train = is_train
        self.use_noisy = False
        self.n_steps = n_steps
        self.use_lstm = True
        self.lstm_size = 512
        self.num_lstm_layers = 1

        if self.restore_model:
            self.epsilon = self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()

        self.replace_target_op = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(self.get_vars('eval_net'), self.get_vars('target_net'))])

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.n_features*2 + self.state_dim*2 + 2))
            self.end_list = [-1]
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)
        #self.sess = tf.Session()
        if self.restore_model == False:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.logger.restore_model(self.sess)

        # self.logger.setup_tf_saver(self.sess, inputs={'image': self.image_input, 'state': self.state_input}, outputs={'q': self.q_eval})

        self.logger.save_model_info(self.sess)

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def _build_net(self, use_fullyconnected = True):
        def build_net(image, state, c_in, h_in, trainable):
            init_state = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            if self.net_type == 'net_b':
                with tf.variable_scope("block_1"):
                    # image_in = tf.transpose(image, [0, 1, 4, 2, 3])  # [batch_size, n_step, n_frame, height, width]
                    # image_in = tf.expand_dims(image_in, -1)  # [batch_size, n_step, n_frame, height, width,1]
                    image_in = tf.reshape(image, [-1, image.shape[2].value, image.shape[3].value,
                                                  image.shape[4].value]) # [batch_size*n_step, height, width, n_frame]
                    image_net = slim.conv2d(image_in, 32, [3, 3], stride=1, scope='conv1', trainable=trainable)
                    image_net = slim.conv2d(image_net, 32, [3, 3], stride=2, scope='conv2', trainable=trainable)
                    #image_net = slim.max_pool2d(image_net, [2, 2], stride=2, scope='max_pool_1')
                with tf.variable_scope("block_2"):
                    image_net = slim.conv2d(image_net, 64, [3, 3], stride=1, scope='conv3', trainable=trainable)
                    image_net = slim.conv2d(image_net, 64, [3, 3], stride=2, scope='conv4', trainable=trainable)
                    #image_net = slim.max_pool2d(image_net, [2, 2], stride=2, scope='max_pool_2')
                with tf.variable_scope("block_3"):
                    image_net = slim.conv2d(image_net, 128, [3, 3], stride=2, scope='conv5', trainable=trainable)
                    #image_net = slim.max_pool2d(image_net, [2, 2], stride=2, scope='max_pool_3')
                    image_net = slim.conv2d(image_net, 128, [3, 3], stride=2, scope='conv6', trainable=trainable)
                    #image_net = slim.max_pool2d(image_net, [2, 2], stride=2, scope='max_pool_4')
                with tf.variable_scope("state"):
                    state_net = tf.layers.dense(state, 128, activation=tf.nn.relu, trainable=trainable)
                    state_net = tf.reshape(state_net, [-1, 1, 1, 128])
                    state_net = tf.tile(state_net, [1, 4, 4, 1])
                with tf.variable_scope("connect"):
                    # net = tf.math.add(image_net, state_net)
                    net = tf.add(image_net, state_net)
                    net = slim.flatten(net)
                    # TODO
                with tf.variable_scope('lstm'):
                    if self.use_lstm:
                        net_shape = net.shape[1].value
                        lstm_input = tf.reshape(net, [-1, self.n_steps, net_shape])
                        cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True)
                        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
                        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_input, initial_state=init_state, time_major=False)
                        # net = tf.reshape(lstm_outputs, shape=[-1, self.n_steps*self.lstm_size])
                        state_output_c = state[0][0]
                        state_output_h = state[0][1]
                        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]))
                        net = lstm_outputs[-1]

                with tf.variable_scope('fc'):
                    if self.use_noisy:
                        w_i = tf.random_uniform_initializer(-0.1, 0.1)
                        b_i = tf.constant_initializer(0.1)

                        net = noisy_dense(net, 512, [512], c_name, c_name_fix, w_i, b_i, name_scope='fc1')
                        net = noisy_dense(net, 512, [512], c_name, c_name_fix, w_i, b_i, name_scope='fc2')
                        out = noisy_dense(net, self.n_actions, [self.n_actions], c_name, c_name_fix, w_i, b_i,
                                          name_scope='logit')
                    else:
                        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, scope='fc1', trainable=trainable)
                        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, scope='fc2', trainable=trainable)
                        out = slim.fully_connected(net, self.n_actions, activation_fn=None, scope='logit',
                                                   trainable=trainable)
            elif self.net_type == 'drqn':
                with tf.variable_scope("block_1"):
                    # image_in = tf.transpose(image, [0, 1, 4, 2, 3])  # [batch_size, n_step, n_frame, height, width]
                    # image_in = tf.expand_dims(image_in, -1)  # [batch_size, n_step, n_frame, height, width,1]
                    image_in = tf.reshape(image, [-1, image.shape[2].value, image.shape[3].value,
                                                  image.shape[4].value])  # [batch_size*n_step, height, width, n_frame]
                    image_net = slim.conv2d(image_in, 32, [8, 8], stride=[4, 4],
                                            activation_fn=tf.nn.leaky_relu, padding='VALID',
                                            scope='conv1', trainable=trainable)
                    image_net = slim.conv2d(image_net, 64, [4, 4], stride=[2, 2],
                                            activation_fn=tf.nn.leaky_relu, padding='VALID',
                                            scope='conv2', trainable=trainable)
                    image_net = slim.conv2d(image_net, 64, [3, 3], stride=[1, 1],
                                            activation_fn=tf.nn.leaky_relu, padding='VALID',
                                            scope='conv3', trainable=trainable)
                with tf.variable_scope("state"):
                    state_net = tf.layers.dense(state, 64, activation=tf.nn.leaky_relu, trainable=trainable)
                    state_net = tf.reshape(state_net, [-1, 1, 1, 64])
                    state_net = tf.tile(state_net, [1, 4, 4, 1])
                with tf.variable_scope("connect"):
                    # net = tf.math.add(image_net, state_net)
                    net = tf.add(image_net, state_net)
                    net = slim.flatten(net)
                    # TODO
                with tf.variable_scope('lstm'):
                    if self.use_lstm:
                        net_shape = net.shape[1].value
                        lstm_input = tf.reshape(net, [-1, self.n_steps, net_shape])
                        cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, activation=tf.nn.leaky_relu, state_is_tuple=True)
                        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
                        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_input, initial_state=init_state,
                                                                      time_major=False)
                        # net = tf.reshape(lstm_outputs, shape=[-1, self.n_steps*self.lstm_size])
                        state_output_c = state[0][0]
                        state_output_h = state[0][1]
                        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]))
                        net = lstm_outputs[-1]
                with tf.variable_scope('fc'):
                    net = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu, scope='fc1',
                                               trainable=trainable)
                    net = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu, scope='fc2',
                                               trainable=trainable)
                    out = slim.fully_connected(net, self.n_actions, activation_fn=None, scope='logit',
                                               trainable=trainable)

            elif self.net_type == 'net_small':
                with tf.variable_scope("block_1"):
                    image_in = tf.reshape(image, [-1, image.shape[2].value, image.shape[3].value,
                                                  image.shape[4].value])  # [batch_size*n_step, height, width, n_frame]
                    net = slim.conv2d(image_in, 64, [3, 3], stride=1, scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='max_pool_1')
                with tf.variable_scope("block_2"):
                    net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='max_pool_2')
                with tf.variable_scope("block_3"):
                    net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='max_pool_3')
                with tf.variable_scope("connect_vw"):
                    net = slim.flatten(net)
                    net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
                    state_vw = tf.reshape(state[:, :, 2:4], [-1, 2])
                    net = tf.concat([net, state_vw], axis=1)
                with tf.variable_scope('lstm'):
                    if self.use_lstm:
                        net_shape = net.shape[1].value
                        lstm_input = tf.reshape(net, [-1, self.n_steps, net_shape])
                        cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, activation=tf.nn.relu, state_is_tuple=True)
                        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
                        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_input, initial_state=init_state,
                                                                      time_major=False)
                        # net = tf.reshape(lstm_outputs, shape=[-1, self.n_steps*self.lstm_size])
                        state_output_c = state[0][0]
                        state_output_h = state[0][1]
                        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]))
                        net = lstm_outputs[-1]
                with tf.variable_scope("connect_goal"):
                    state_goal = tf.layers.dense(net, units=512, activation=tf.nn.relu)
                    state_vw = tf.reshape(state[:, self.n_steps-1, 0:2], [-1, 2])
                    net = tf.concat([net, state_goal], axis=1)
                with tf.variable_scope("fc"):
                    net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
                    net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
                    out = tf.layers.dense(net, units=self.n_actions, activation=None)

            if self.use_lstm:
                return out, state_output_c, state_output_h
            else:
                return out

        # ------------------ build evaluate_net ------------------
        self.image_input = tf.placeholder(tf.float32,
                                          [None, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]],
                                          name='image_input')  # input
        self.state_input = tf.placeholder(tf.float32,
                                          [None, self.n_steps, self.state_dim],
                                          name='state_input')
        self.q_target = tf.placeholder(tf.float32,
                                       [None, self.n_actions],
                                       name='Q_target')  # for calculating loss
        # tmp_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True)
        # self.lstm_state_in = tmp_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.c_state_in = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        self.h_state_in = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        self.c_state_out = tf.placeholder(tf.float32, [None, self.lstm_size], name="out_c")
        self.h_state_out = tf.placeholder(tf.float32, [None, self.lstm_size], name="out_h")
        # self.lstm_state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            self.q_eval, self.c_state_out, self.h_state_out = \
                build_net(self.image_input, self.state_input, self.c_state_in, self.h_state_in, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.image_input_ = tf.placeholder(tf.float32,
                                           [None, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]],
                                           name='image_input_')  # input
        self.state_input_ = tf.placeholder(tf.float32,
                                           [None, self.n_steps, self.state_dim],
                                           name='state_input_')
        # self.c_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        # self.h_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")
        # self.lstm_state_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_target, self.h_state_target)
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next, self.c_state_out, self.h_state_out = \
                build_net(self.image_input_, self.state_input_, self.c_state_in, self.h_state_in, False)

    def store_transition(self, s, a, r, s_, is_done):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s[1].flatten(), s[0], [a, r], s_[0], s_[1].flatten()))
            self.memory.store(transition, self.n_features, self.state_dim, is_done)    # have high priority for newly arrived transition
        else:       # random replay
            transition = np.hstack((s[1].flatten(), s[0], [a, r], s_[0], s_[1].flatten()))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            if is_done:
                self.end_list.append(index)
                # 当memory存满时，删除被覆盖的episode所对应的index
                if self.memory_counter >= self.memory_size:
                    if self.end_list[0] == -1:
                        del self.end_list[0]
                    ep_head = self.end_list[-2]+1
                    ep_tail = index
                    while ((self.end_list[0] >= ep_head) and (self.end_list[0] <= ep_tail)) or \
                            ((ep_head > ep_tail) and ((self.end_list[0] >= ep_head) or (self.end_list[0] <= ep_tail))):
                        del self.end_list[0]
            self.memory_counter += 1

    def choose_action(self, observation, actions, c_state_in, h_state_in, is_test=False, use_vfh=False):
        image_input = observation[1]
        state_input = observation[0]
        image_input = image_input[np.newaxis, :]
        state_input = state_input[np.newaxis, :]
        if is_test:
            if use_vfh:
                scan = observation[2]
                action = vfh_plan(state_input[0], scan, 'kejiarobot0')
                print "test vfh output: ", action
                action = get_close_vw(action, actions)
            else:
                actions_value = self.sess.run([self.q_eval], feed_dict={
                    self.image_input: image_input,
                    self.state_input: state_input,
                    self.c_state_in: c_state_in,
                    self.h_state_in: h_state_in
                })
                action = np.argmax(actions_value)
                print "is_test    ", action
            return action
        else:
            if np.random.uniform() < self.epsilon:
                actions_value = self.sess.run([self.q_eval], feed_dict={
                    self.image_input: image_input,
                    self.state_input: state_input,
                    self.c_state_in: c_state_in,
                    self.h_state_in: h_state_in
                })
                action = np.argmax(actions_value)
                #print('model action:', action)
            else:
                if use_vfh:
                    if np.random.rand()<0.15:
                        action = np.random.randint(0, self.n_actions)
                        print('random action:', action)
                    else:
                        scan = observation[2]
                        action = vfh_plan(state_input[0], scan, 'kejiarobot0')
                        print "vfh output: ", action
                        action = get_close_vw(action, actions)
                        print('vfh action:', action)
                    return action
                else:
                    action = np.random.randint(0, self.n_actions)
            return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)#TODO
        else:
            sample_index = np.random.choice(len(self.end_list)-1, size=self.batch_size)
            batch_memory = np.empty((self.batch_size, self.n_steps, self.n_features*2 + self.state_dim*2 + 2))
            for i in range(self.batch_size):
                a = self.end_list[sample_index[i]]
                b = self.end_list[sample_index[i]+1]
                if a < b:
                    random_step = np.random.randint(a+1, b+2-self.n_steps)
                    batch_memory[i, :, :] = self.memory[random_step:random_step+self.n_steps]
                else:
                    random_step = np.random.randint(a+1, b+2-self.n_steps+self.memory_size)
                    for j in range(self.n_steps):
                        batch_memory[i, j] = self.memory[(random_step+j) % self.memory_size]

        c_state_batch = np.zeros((self.batch_size, self.lstm_size), dtype=np.float32)
        h_state_batch = np.zeros((self.batch_size, self.lstm_size), dtype=np.float32)

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.image_input_: batch_memory[:, :, -self.n_features:].reshape(self.batch_size, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]),
                           self.state_input_: batch_memory[:, :, -self.n_features-self.state_dim:-self.n_features],
                           self.image_input: batch_memory[:, :, :self.n_features].reshape(self.batch_size, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]),
                           self.state_input: batch_memory[:, :, self.n_features:self.n_features+self.state_dim],
                           self.c_state_in: c_state_batch,
                           self.h_state_in: h_state_batch})

        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, -1, self.n_features+self.state_dim].astype(int)
        reward = batch_memory[:, -1, self.n_features+self.state_dim + 1]
        max_q_next = np.max(q_next, axis=1)
        for i in range(self.batch_size):
            q_target[i, eval_act_index[i]] = reward[i] + self.gamma*max_q_next[i]

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.image_input: batch_memory[:, :, :self.n_features].reshape(self.batch_size, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                     self.state_input: batch_memory[:, :, self.n_features:self.n_features+self.state_dim],
                                                     self.c_state_in: c_state_batch,
                                                     self.h_state_in: h_state_batch,
                                                     self.q_target: q_target,
                                                     self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.image_input: batch_memory[:, :, :self.n_features].reshape(self.batch_size, self.n_steps, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                    self.state_input: batch_memory[:, :, self.n_features:self.n_features+self.state_dim],
                                                    self.c_state_in: c_state_batch,
                                                    self.h_state_in: h_state_batch,
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        if self.prioritized:
            return abs_errors, self.cost
        else:
            return self.cost

    def save_train_model(self):
        self.logger.save_train_model(self.sess)

    def freeze_graph(self):
        self.logger.freezing_graph("eval_net/fc/dense_2")
