from deeprl_prj.policy import *
from deeprl_prj.objectives import *
from deeprl_prj.preprocessors import *
from deeprl_prj.utils import *
from deeprl_prj.core import *
from helper import *
from tqdm import tqdm
import numpy as np
from tensorflow.python.ops.distributions.normal import Normal
import sys
from gym import wrappers
import tensorflow as tf
import cv2
video = []
count = 500
"""Main DQN agent."""
def print_param():
    from functools import reduce
    from operator import mul
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("all param:", num_params)


class attention_conv:
    def __init__(self):
        pass

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def at_flatten(self, x):
        return tf.reshape(x, shape=[-1, 49, x.shape[-1]])

    def attention(self, x):
        with tf.variable_scope('att', reuse=tf.AUTO_REUSE):

            w_init = tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(0, 1e-2)
            b_init = tf.constant_initializer(0.0)            # tf.constant_initializer(1e-2)


            self.conv1 = tf.contrib.layers.convolution2d( \
                inputs=x, num_outputs=512, \
                kernel_size=[1, 1], stride=[1, 1], padding='VALID', \
                activation_fn=tf.nn.elu, biases_initializer=b_init, scope='_conv1')

            self.conv2 = tf.contrib.layers.convolution2d( \
                inputs=self.conv1, num_outputs=1, \
                kernel_size=[1, 1], stride=[1, 1], padding='VALID', \
                activation_fn=tf.nn.elu, biases_initializer=b_init, scope='_conv2')


            g = tf.nn.softmax( tf.reshape(self.conv2,[-1,49,1]),dim=1)  # (32, 49, 2)
            g = tf.transpose(tf.reshape(g, [-1, 7,7, 1]),[0,3,1,2]) # (-1,1, 7, 7)

            return g

class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, myScope):
        #if self.args.model == 'GMAQN':

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)
        self.imageIn =  tf.placeholder(shape=[None,84,84,num_frames],dtype=tf.float32)
        self.args = args

        w_init = tf.contrib.layers.xavier_initializer() #tf.truncated_normal_initializer(0, 2e-2)
        b_init = tf.constant_initializer(0.0)
        with tf.variable_scope(myScope+'conv1'): #img [32,84,84,4]
            w_conv1 = tf.get_variable('w_conv1', [8, 8, num_frames, 32], initializer=w_init)
            b_conv1 = tf.get_variable('b_conv1', [32], initializer=b_init)
            conv1 = self.conv2d(self.imageIn, w_conv1, 4)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1)) #(?, 20, 20, 32)

        with tf.variable_scope(myScope+'conv2'):
            w_conv2 = tf.get_variable('w_conv2', [4, 4, 32, 64], initializer=w_init)
            b_conv2 = tf.get_variable('b_conv2', [64], initializer=b_init)
            conv2 = self.conv2d(h_conv1, w_conv2, 2)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2)) #(?, 9, 9, 64)

        with tf.variable_scope(myScope+'conv3'):
            w_conv3 = tf.get_variable('w_conv3', [3, 3, 64, 64], initializer=w_init)
            b_conv3 = tf.get_variable('b_conv3', [64], initializer=b_init)
            conv3 = self.conv2d(h_conv2, w_conv3, 1)
            self.h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3)) #(?, 7, 7, 64)

        with tf.variable_scope(myScope+'model'):

            if self.args.model=='GMAQN':
                self.s = tf.nn.l2_normalize(self.h_conv3, dim=3, epsilon=1e-12, name='ln')

                layers = []
                self.h = tf.get_variable('C', [32, 64, 7, 7], initializer=b_init, trainable=False)
                for i in range(3):
                    with tf.variable_scope('att{}'.format(i)):
                        att = attention_conv()
                        layers.append(att.attention(self.s))
                self.s = tf.transpose(self.s, [0, 3, 1, 2])
                f = tf.sigmoid(layers[0] * self.s) * self.h                                 # forget [32,64,7,7]
                i = tf.tanh(layers[1] * self.s) * tf.sigmoid(layers[2] * self.s)  # input [32,1,7,7]
                self.h = f + i
                print_param()
            if self.args.model=='ALSTM':
                self.H = 98
                lstm_cell_att = tf.nn.rnn_cell.LSTMCell(num_units=self.H)
                c_att, h_att = lstm_cell_att.zero_state(self.batch_size, tf.float32) #32 98

                x = tf.nn.l2_normalize(self.h_conv3, dim=-1, epsilon=1e-12, name='ln')

                w_conv1 = tf.get_variable('w_conv1', [1, 1, 64, 512], initializer=w_init)
                b_conv1 = tf.get_variable('b_conv1', [512], initializer=b_init)
                conv1 = self.conv2d(x, w_conv1, 1)
                l1 = tf.nn.elu(tf.nn.bias_add(conv1, b_conv1))

                w_conv2 = tf.get_variable('w_conv2', [1, 1, 512, 2], initializer=w_init)
                b_conv2 = tf.get_variable('b_conv2', [2], initializer=b_init)
                conv2 = self.conv2d(l1, w_conv2, 1)  # 32.7.7.2
                l2 = tf.nn.softmax(tf.reshape(tf.nn.bias_add(conv2, b_conv2), [-1, 49, 2]), dim=1)  # 32.7.7.2
                l2 = tf.reshape(l2, [-1, 7, 7, 2])
                l2 = tf.transpose(l2, [0, 3, 1, 2])  # 32.2,7.7

                with tf.variable_scope(myScope + '_lstmCell'):
                    context = tf.reshape(l2, [-1, 7 * 7 * 2])  
                    _, (c_att, h_att) = lstm_cell_att(inputs=context, state=[c_att, h_att])

                context = tf.reshape(h_att, [-1, 2, 7, 7])
                x = tf.transpose(x, [0, 3, 1, 2])

                x1 = x * tf.expand_dims(context[:, 0, :, :], dim=1)
                x2 = x * tf.expand_dims(context[:, 1, :, :], dim=1)
                x = x1 + x2  # 32.64.7.7
                self.h = tf.transpose(x, [0, 2, 3, 1])  # 32.64.7.7
            if self.args.model=='RS-DQN':
                x = tf.nn.l2_normalize(self.h_conv3,dim=-1,epsilon=1e-12,name='ln')

                w_conv1 = tf.get_variable('w_conv1', [1, 1, 64, 512], initializer=w_init)
                b_conv1 = tf.get_variable('b_conv1', [512], initializer=b_init)
                conv1 = self.conv2d(x, w_conv1, 1)
                l1 = tf.nn.elu(tf.nn.bias_add(conv1, b_conv1))

                w_conv2 = tf.get_variable('w_conv2', [1, 1, 512, 2], initializer=w_init)
                b_conv2 = tf.get_variable('b_conv2', [2], initializer=b_init)
                conv2 = self.conv2d(l1, w_conv2, 1) #32.7.7.2
                l2 = tf.nn.softmax(tf.reshape(tf.nn.bias_add(conv2, b_conv2),[-1,49,2]),dim=1)  #32.7.7.2
                l2  = tf.reshape(l2,[-1,7,7,2])
                l2 = tf.transpose(l2,[0,3,1,2]) #32.2,7.7

                x = tf.transpose(x,[0,3,1,2])

                x1 = x * tf.expand_dims(l2[:,0,:,:],dim=1)
                x2 = x * tf.expand_dims(l2[:,1,:,:],dim=1)
                x = x1 + x2 #32.64.7.7
                self.h = tf.transpose(x, [0,2,3,1])  # 32.64.7.7
            if self.args.model=='Local':

                self.L = 7 * 7
                self.D = 64
                self.H = 256
                self.channel = 64
                self.loc_dim = 5
                self.img_size = 7
                self.pth_size = 3
                self.num_glimpses = 4
                self.glimpse_output_size = 256
                self.cell_size = 256
                self.weight_initializer = tf.contrib.layers.xavier_initializer()
                self.const_initializer = tf.constant_initializer(0.0)
                self.alpha_list = []

                # lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.H)
                # c, h = lstm_cell.zero_state(self.batch_size, tf.float32)

                lstm_cell_att = tf.nn.rnn_cell.LSTMCell(num_units=self.H)
                c_att, h_att = lstm_cell_att.zero_state(self.batch_size, tf.float32)

                for t in range(self.num_glimpses):
                    loc = self._attention_layer(h_att, myScope=myScope, reuse=(t != 0))  # t=0，the first Scope
                    # print(loc.shape)    (?, 5)
                    # input(2)
                    # feature and loc input to GlimpseNetwork
                    context = self.GlimpseNetwork(self.h_conv3, loc, myScope=myScope)  # conv3 (None,  7, 7, 64)
                    # print(context.shape)  #(n,576)
                    with tf.variable_scope(myScope + '_lstmCell', reuse=(t != 0)):
                        _, (c_att, h_att) = lstm_cell_att(inputs=context, state=[c_att, h_att])  # (?, 256) (?, 256)
                self.convFlat = tf.reshape(h_att, [self.batch_size, 256])
            if self.args.model=='DQN' :
                self.h = self.h_conv3 #32,7,7,64
        if self.args.model != 'Local':
            self.convFlat = tf.reshape(self.h,[self.batch_size, 49*64])

        self.rnn = tf.contrib.layers.fully_connected(self.convFlat, h_size, activation_fn=tf.nn.relu)

        self.Qout = tf.contrib.layers.fully_connected(self.rnn, num_actions, activation_fn=None) # (32,18)
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        gradients = self.optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        self.updateModel = self.optimizer.apply_gradients(capped_gradients)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def RetinaSensor(self, normLoc):
        A = B = self.img_size  # 7
        N = self.pth_size  # 3
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(normLoc, 5, 1)
        gx = (A + 1) / 2 * (gx_ + 1)
        gy = (B + 1) / 2 * (gy_ + 1)
        sigma2 = tf.exp(log_sigma2)
        # sigma2 -= 0.9
        delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N    delta:步长

        return filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma),)

    def GlimpseNetwork(self, feature_map, locs, myScope=''):
        B = A = self.img_size  # 7
        N = self.pth_size  # 3
        Fx, Fy, gamma = self.RetinaSensor(locs)

        Fxt = tf.transpose(Fx, perm=[0, 2, 1])  # [?, 7, 3]

        feature_map = tf.transpose(feature_map, perm=[0, 3, 1, 2])  # [?, 64, 7, 7]
        img = tf.reshape(feature_map, [-1, 64 * B, A])

        img_Fxt = tf.matmul(img, Fxt)  # [?, 64*7, 3]                  [?, 7, 3, 64]
        img_Fxt = tf.reshape(tf.transpose(tf.reshape(img_Fxt, [-1, 64, 7, 3]), [0, 2, 3, 1]), [-1, 7, 3 * 64])
        glimpse = tf.matmul(Fy, img_Fxt)

        glimpse = tf.reshape(glimpse, [-1, N * N * 64])
        # print(gamma.shape)  (?, 9)

        x = glimpse * tf.reshape(gamma, [-1, 1])  # batch x (read_n*read_n)
        # print(x.shape)   # (?, 576)
        # print(glimpse.shape)  # (?, 576)
        return glimpse
    def _attention_layer(self, h, myScope, reuse=False):
        with tf.variable_scope(myScope + '_attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.loc_dim], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.loc_dim], initializer=self.const_initializer)

            loc = tf.matmul(h, w) + b  # (N,5)
            return loc
eps = 1e-8
def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    #gx += 12
    #gy += 12
    A = B = 7
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
    # normalize, sum over A and B dims
    Fx = Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy = Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx, Fy
def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the class 
    provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self, args, num_actions):
        self.args = args
        self.num_actions = num_actions
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)   
        self.atari_processor = AtariPreprocessor()   
        self.memory = ReplayMemory(args)
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon, args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate_ADM
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.output_path_videos = args.output + '/videos/'
        self.output_path_images = args.output + '/images/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.model

        self.h_size = 512
        self.tau = 0.001
        tf.reset_default_graph()
        #We define the cells for the primary and target q-networks
        #cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        #cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        self.q_network = Qnetwork(args=args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, myScope="QNet")
        self.target_network = Qnetwork(args=args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, myScope="TargetNet")
        
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        trainables = tf.trainable_variables()

        self.targetOps = updateTargetGraph(trainables, self.tau)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        updateTarget(self.targetOps, self.sess)
        self.writer = tf.summary.FileWriter(self.output_path)
        if args.restore:
            self.restore_model(args.restore_path)

        print(self.get_num_params())

    def get_num_params(self):
        from functools import reduce
        from operator import mul
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    def calc_q_values(self, t,rgb,s):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if self.args.model=='GMAQN':
            state = np.zeros([32, 84, 84, self.num_frames])
            state[0, :, :, :] = s
            ############ do a step ############
            Qout = self.sess.run(self.q_network.Qout,feed_dict={self.q_network.imageIn: state,self.q_network.batch_size:32})
            Qout = np.expand_dims(Qout[0], axis=0)
        else:
            state = s[None, :, :, :]
            Qout = self.sess.run(self.q_network.Qout,feed_dict={self.q_network.imageIn: state, self.q_network.batch_size: 1})
        return Qout

    def select_action(self, t,rgbstate,state, is_training = True, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(t,rgbstate,state)
        if is_training:
            if kwargs['policy_type'] == 'UniformRandomPolicy':
                return UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                return self.policy.select_action(q_values, is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            return GreedyPolicy().select_action(q_values)

    def update_policy(self, current_sample):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch_size = self.batch_size

        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = self.memory.sample(batch_size)
            samples = self.atari_processor.process_batch(samples)

            states = np.stack([x.state for x in samples])
            actions = np.asarray([x.action for x in samples])
            # action_mask = np.zeros((batch_size, self.num_actions))
            # action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            # next_qa_value = self.target_network.predict_on_batch(next_states)
            next_qa_value = self.sess.run(self.target_network.Qout,
                                          feed_dict={self.target_network.imageIn: next_states,
                                                     self.target_network.batch_size: batch_size})
        if self.enable_ddqn:
            # qa_value = self.q_network.predict_on_batch(next_states)
            qa_value = self.sess.run(self.q_network.Qout,
                                     feed_dict={self.q_network.imageIn: next_states,
                                                self.q_network.batch_size: batch_size})
            max_actions = np.argmax(qa_value, axis=1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis=1)
        # print rewards.shape, mask.shape, next_qa_value.shape, batch_size
        target = rewards + self.gamma * mask * next_qa_value

        loss, _, rnn = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn],
                                     feed_dict={self.q_network.imageIn: states, self.q_network.batch_size: batch_size,
                                                self.q_network.actions: actions, self.q_network.targetQ: target})
        return loss, np.mean(target)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        self.is_training = True
        print("Training starts.")
        # self.save_model(0)
        eval_count = 0

        state = env.reset()
        burn_in = True
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0
        count = 0
        patience = 20
        max_episode_reward_mean = -1
        num_game, ep_reward = 0, 0.
        ep_rewards, ep_loss, ep_target_value, actions = [], [], [], []
        for t in tqdm(range(self.num_burn_in + num_iterations)):
            if t == self.num_burn_in:
                num_game,  ep_reward = 0, 0.
                ep_rewards, ep_loss,ep_target_value = [], [],[]

            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"

            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            action = self.select_action(t,state, action_state, self.is_training, policy_type=policy_type)
            processed_state = self.atari_processor.process_state_for_memory(state)
            state, reward, done, info = env.step(action)
            if self.args.noise != 'none':
                state = self.add_noise(state,env)  # add noise

            processed_next_state = self.atari_processor.process_state_for_network(state)
            action_next_state = np.dstack((action_state, processed_next_state))
            action_next_state = action_next_state[:, :, 1:]

            processed_reward = self.atari_processor.process_reward(reward)

            self.memory.append(processed_state, action, processed_reward, done)
            current_sample = Sample(action_state, action, processed_reward, action_next_state, done)

            if not burn_in:
                episode_frames += 1
                episode_raw_reward += reward
                if episode_frames > max_episode_length:
                    done = True

            if done:

                last_frame = self.atari_processor.process_state_for_memory(state)
                self.memory.append(last_frame, action, 0, done)
                if not burn_in:
                    avg_target_value = episode_target_value / episode_frames
                    episode_frames = 0
                    num_game += 1
                    ep_rewards.append(episode_raw_reward)
                    ep_loss.append(episode_loss)
                    ep_target_value.append(avg_target_value)

                    episode_raw_reward = .0
                    episode_loss = .0
                    episode_target_value = .0
                    idx_episode += 1


                burn_in = (t < self.num_burn_in)
                state = env.reset()
                self.atari_processor.reset()
                self.history_processor.reset()

            if not burn_in:
                if t %  10000 ==0:
                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                        avg_ep_loss=np.mean(episode_loss)
                        avg_ep_q=np.mean(episode_target_value)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward,avg_ep_loss,avg_ep_q = 0, 0, 0,0,0
                    print(
                        '\nTrain:ep %d, avg_ep_r: %.4f, min_ep_r: %.4f, max_ep_r: %.4f, avg_ls: %.6f, avg_q: %3.6f,# game: %d' \
                        % ( idx_episode,avg_ep_reward,  min_ep_reward,max_ep_reward,avg_ep_loss, avg_ep_q,  num_game))
                    with open( './data/'+self.args.model +'_' + self.args.noise + '_'+ self.args.env + '.txt', 'a') as file:
                        file.write(str(avg_ep_reward) + '\n')
                    num_game = 0
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                if t % self.train_freq == 0:
                    loss, target_value = self.update_policy(current_sample)
                    episode_loss += loss
                    episode_target_value += target_value
                # update freq is based on train_freq
                if t % self.target_update_freq == 0:
                    updateTarget(self.targetOps, self.sess)
                    print("----- Synced.")
                if t % self.save_freq == 0:
                    self.save_model(idx_episode)
                if t % (self.eval_freq * self.train_freq) == 0:
                    pass
                    #episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 10, eval_count, max_episode_length, True)
            
    def add_noise(self,state,env):
        import cv2
        VP_W  = 210
        VP_H = 160
        S = {'noise1':50,'noise2':80, 'noise3':128}
        rand_noise = np.random.randint(0, S[self.args.noise], VP_W * VP_H * 3)
        rand_noise = rand_noise.reshape([ VP_W, VP_H, 3])  #210 160 3
        state = np.uint8(rand_noise + state)
        # env.render()
        # im  = np.dstack((
        #     state[:, :, 2],
        #     state[:, :, 1],
        #     state[:, :, 0],
        # ))
        # cv2.imshow("Image", im)
        return state
    def save_model(self, idx_episode):
        safe_path = self.output_path + "/qnet" + str(idx_episode) + ".cptk"
        self.saver.save(self.sess, safe_path)
        print("Network at", idx_episode, "saved to:", safe_path)

    def restore_model(self, path):
        model_file = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, model_file)
        print("+++++++++ Network restored from: %s", model_file)

    def evaluate(self, env, num_episodes, eval_count, max_episode_length=None, monitor=True):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("Evaluation starts.")
        # plt.figure(1, figsize=(40, 20))

        is_training = False
        
        state = env.reset()

        idx_episode = 1
        episode_frames = 0
        episode_reward = np.zeros(num_episodes)
        t = 0

        while idx_episode <= num_episodes:
            t += 1
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            action = self.select_action(action_state, is_training, policy_type='GreedyEpsilonPolicy')

            state, reward, done, info = env.step(action)
            episode_frames += 1
            episode_reward[idx_episode - 1] += reward
            if episode_frames > max_episode_length:
                done = True
            if done:
                print("Eval: time %d, episode %d, length %d, reward %.0f. @eval_count %s" %
                      (t, idx_episode, episode_frames, episode_reward[idx_episode - 1], eval_count))
                eval_count += 1

                state = env.reset()
                episode_frames = 0
                idx_episode += 1
                self.atari_processor.reset()
                self.history_processor.reset()

        reward_mean = np.mean(episode_reward)
        reward_std = np.std(episode_reward)
        with open('reward_mean_common_dqn_seaquest_2frame.txt','a') as file:
            file.write(str(reward_mean)+'\n')
        print("Evaluation summury: num_episodes [%d], reward_mean [%.3f], reward_std [%.3f]" %
            (num_episodes, reward_mean, reward_std))
        #sys.stdout.flush()

        return reward_mean, reward_std, eval_count
