import datetime,gym,os,pybullet_envs,psutil,time,os
import scipy.signal
import numpy as np
import tensorflow as tf

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x 
    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    if with_min_and_max:
        global_min = (np.min(x) if len(x) > 0 else np.inf)
        global_max = (np.max(x) if len(x) > 0 else -np.inf)
        return mean, std, global_min, global_max
    return mean, std

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input: 
        vector x, [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, odim, adim, size=5000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, adim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]
        
def create_ppo_model(env=None,hdims=[256,256],output_actv=None):
    """
    Create PPO Actor-Critic Model (compatible with Ray)
    """
    import tensorflow as tf # make it compatible with Ray actors
    from gym.spaces import Box, Discrete
    
    def mlp(x, hdims=[64,64], actv=tf.nn.relu, output_actv=None):
        for h in hdims[:-1]:
            x = tf.layers.dense(x, units=h, activation=actv)
        return tf.layers.dense(x, units=hdims[-1], activation=output_actv)
    
    def mlp_categorical_policy(o, a, hdims=[64,64], actv=tf.nn.relu, output_actv=None, action_space=None):
        adim = action_space.n
        logits = mlp(x=o, hdims=hdims+[adim], actv=actv, output_actv=None)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
        logp = tf.reduce_sum(tf.one_hot(a, depth=adim) * logp_all, axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=adim) * logp_all, axis=1)
        return pi, logp, logp_pi, pi
    
    def gaussian_likelihood(x, mu, log_std):
        EPS = 1e-8
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    
    def mlp_gaussian_policy(o, a, hdims=[64,64], actv=tf.nn.relu, output_actv=None, action_space=None):
        adim = a.shape.as_list()[-1]
        mu = mlp(x=o, hdims=hdims+[adim], actv=actv, output_actv=output_actv)
        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(adim, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        return pi, logp, logp_pi, mu # <= mu is added for the deterministic policy
    
    def mlp_actor_critic(o, a, hdims=[64,64], actv=tf.nn.relu, 
                     output_actv=None, policy=None, action_space=None):
        if policy is None and isinstance(action_space, Box):
            policy = mlp_gaussian_policy
        elif policy is None and isinstance(action_space, Discrete):
            policy = mlp_categorical_policy

        with tf.variable_scope('pi'):
            pi, logp, logp_pi, mu = policy(
                o=o, a=a, hdims=hdims, actv=actv, output_actv=output_actv, action_space=action_space)
        with tf.variable_scope('v'):
            v = tf.squeeze(mlp(x=o, hdims=hdims+[1], actv=actv, output_actv=None), axis=1)
        return pi, logp, logp_pi, v, mu
    
    def placeholder(dim=None):
        return tf.placeholder(dtype=tf.float32,shape=(None,dim) if dim else (None,))
    
    def placeholders(*args):
        """
        Usage: a_ph,b_ph,c_ph = placeholders(adim,bdim,None)
        """
        return [placeholder(dim) for dim in args]
    
    def get_vars(scope):
        return [x for x in tf.compat.v1.global_variables() if scope in x.name]
    
    # Have own session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Placeholders
    odim = env.observation_space.shape[0]
    adim = env.action_space.shape[0]
    o_ph,a_ph,adv_ph,ret_ph,logp_old_ph = placeholders(odim,adim,None,None,None)
    
    # Actor-critic model 
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    actor_critic = mlp_actor_critic
    pi,logp,logp_pi,v,mu = actor_critic(o_ph, a_ph, **ac_kwargs)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [o_ph, a_ph, adv_ph, ret_ph, logp_old_ph]
    
    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    
    # Get variables
    pi_vars,v_vars = get_vars('pi'),get_vars('v')
    
    # Accumulate model
    model = {'o_ph':o_ph,'a_ph':a_ph,'adv_ph':adv_ph,'ret_ph':ret_ph,'logp_old_ph':logp_old_ph,
             'pi':pi,'logp':logp,'logp_pi':logp_pi,'v':v,'mu':mu,
             'all_phs':all_phs,'get_action_ops':get_action_ops,'pi_vars':pi_vars,'v_vars':v_vars}
    return model,sess

def create_ppo_graph(model,clip_ratio=0.2,pi_lr=3e-4,vf_lr=1e-3,epsilon=1e-2):
    """
    Create PPO Graph
    """
    # PPO objectives
    ratio = tf.exp(model['logp'] - model['logp_old_ph']) # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(model['adv_ph']>0,
                       (1+clip_ratio)*model['adv_ph'], (1-clip_ratio)*model['adv_ph'])
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * model['adv_ph'], min_adv))
    v_loss = tf.reduce_mean((model['ret_ph'] - model['v'])**2)
    
    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(model['logp_old_ph'] - model['logp']) # a sample estimate for KL-divergence
    approx_ent = tf.reduce_mean(-model['logp']) # a sample estimate for entropy
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
    
    # Optimizers
    pi_ent_loss = pi_loss - 0.01 * approx_ent # entropy-reg policy loss 
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr,epsilon=epsilon).minimize(pi_ent_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr,epsilon=epsilon).minimize(v_loss)
    
    # Accumulate graph
    graph = {'pi_loss':pi_loss,'v_loss':v_loss,'approx_kl':approx_kl,'approx_ent':approx_ent,
             'clipfrac':clipfrac,'train_pi':train_pi,'train_v':train_v}
    return graph


def update_ppo(model,graph,sess,buf,train_pi_iters=100,train_v_iters=100,target_kl=0.01):
    """
    Update PPO
    """
    feeds = {k:v for k,v in zip(model['all_phs'], buf.get())}
    pi_l_old, v_l_old, ent = sess.run(
        [graph['pi_loss'],graph['v_loss'],graph['approx_ent']],feed_dict=feeds)
    # Training
    for i in range(train_pi_iters):
        _, kl = sess.run([graph['train_pi'],graph['approx_kl']],feed_dict=feeds)
        if kl > 1.5 * target_kl:
            break
    for _ in range(train_v_iters):
        sess.run(graph['train_v'],feed_dict=feeds)
    # Log changes from update
    pi_l_new,v_l_new,kl,cf = sess.run(
        [graph['pi_loss'],graph['v_loss'],graph['approx_kl'],graph['clipfrac']],
        feed_dict=feeds)
    

def save_ppo_model(npz_path,R,VERBOSE=True):
    """
    Save PPO model weights
    """
    
    # PPO model
    tf_vars = R.model['pi_vars'] + R.model['v_vars']
    data2save,var_names,var_vals = dict(),[],[]
    for v_idx,tf_var in enumerate(tf_vars):
        var_name,var_val = tf_var.name,R.sess.run(tf_var)
        var_names.append(var_name)
        var_vals.append(var_val)
        data2save[var_name] = var_val
        if VERBOSE:
            print ("[%02d]  var_name:[%s]  var_shape:%s"%
                (v_idx,var_name,var_val.shape,)) 
    
    # Create folder if not exist
    dir_name = os.path.dirname(npz_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print ("[%s] created."%(dir_name))
        
    # Save npz
    np.savez(npz_path,**data2save)
    print ("[%s] saved."%(npz_path))
    

def restore_ppo_model(npz_path,R,VERBOSE=True):
    """
    Restore PPO model weights
    """
    
    # Load npz
    l = np.load(npz_path)
    print ("[%s] loaded."%(npz_path))
    
    # Get values of PPO model  
    tf_vars = R.model['pi_vars'] + R.model['v_vars']
    var_vals = []
    for tf_var in tf_vars:
        var_vals.append(l[tf_var.name])   
        
    # Assign weights of PPO model
    R.set_weights(var_vals)
    