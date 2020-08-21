import os,ray
import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, odim, adim, size):
        self.obs1_buf = np.zeros([size, odim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, odim], dtype=np.float32)
        self.acts_buf = np.zeros([size, adim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
        
    def get(self):
        names = ['obs1_buf','obs2_buf','acts_buf','rews_buf','done_buf',
                 'ptr','size','max_size']
        vals =[self.obs1_buf,self.obs2_buf,self.acts_buf,self.rews_buf,self.done_buf,
               self.ptr,self.size,self.max_size]
        return names,vals

    def restore(self,a):
        self.obs1_buf = a[0]
        self.obs2_buf = a[1]
        self.acts_buf = a[2]
        self.rews_buf = a[3]
        self.done_buf = a[4]
        self.ptr = a[5]
        self.size = a[6]
        self.max_size = a[7]
        
def create_sac_model(odim=10,adim=2,hdims=[256,256],actv=tf.nn.relu):
    """
    Soft Actor Critic Model (compatible with Ray)
    """
    import tensorflow as tf # make it compatible with Ray actors
    
    def mlp(x,hdims=[256,256],actv=tf.nn.relu,out_actv=tf.nn.relu):
        ki = tf.truncated_normal_initializer(stddev=0.1)
        # ki = tf.glorot_normal_initializer()
        for hdim in hdims[:-1]:
            x = tf.layers.dense(x,units=hdim,activation=actv,kernel_initializer=ki)
        return tf.layers.dense(x,units=hdims[-1],activation=out_actv,kernel_initializer=ki)
    def gaussian_loglik(x,mu,log_std):
        EPS = 1e-8
        pre_sum = -0.5*(
            ( (x-mu)/(tf.exp(log_std)+EPS) )**2 +
            2*log_std + np.log(2*np.pi)
        )
        return tf.reduce_sum(pre_sum, axis=1)
    def mlp_gaussian_policy(o,adim=2,hdims=[256,256],actv=tf.nn.relu):
        net = mlp(x=o,hdims=hdims,actv=actv,out_actv=actv) # feature 
        mu = tf.layers.dense(net,adim,activation=None) # mu
        log_std = tf.layers.dense(net,adim,activation=None) # log_std
        LOG_STD_MIN,LOG_STD_MAX = -10.0,+2.0
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX) 
        # std = tf.exp(log_std) # std 
        std = tf.sigmoid(log_std) # std 
        pi = mu + tf.random_normal(tf.shape(mu)) * std  # sampled
        logp_pi = gaussian_loglik(x=pi,mu=mu,log_std=log_std) # log lik
        return mu,pi,logp_pi
    def squash_action(mu,pi,logp_pi):
        # Squash those unbounded actions
        logp_pi -= tf.reduce_sum(2*(np.log(2) - pi -
                                    tf.nn.softplus(-2*pi)), axis=1)
        mu,pi = tf.tanh(mu),tf.tanh(pi)
        return mu, pi, logp_pi
    def mlp_actor_critic(o,a,hdims=[256,256],actv=tf.nn.relu,out_actv=None,
                         policy=mlp_gaussian_policy):
        adim = a.shape.as_list()[-1]
        with tf.variable_scope('pi'): # policy
            mu,pi,logp_pi = policy(o=o,adim=adim,hdims=hdims,actv=actv)
            mu,pi,logp_pi = squash_action(mu=mu,pi=pi,logp_pi=logp_pi)
        def vf_mlp(x): return tf.squeeze(
            mlp(x=x,hdims=hdims+[1],actv=actv,out_actv=None),axis=1)
        with tf.variable_scope('q1'): q1 = vf_mlp( tf.concat([o,a],axis=-1))
        with tf.variable_scope('q2'): q2 = vf_mlp( tf.concat([o,a],axis=-1))
        return mu,pi,logp_pi,q1,q2
    
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
    o_ph,a_ph,o2_ph,r_ph,d_ph = placeholders(odim,adim,odim,None,None)
    # Actor critic 
    ac_kwargs = {'hdims':hdims,'actv':actv,'out_actv':None,'policy':mlp_gaussian_policy}
    with tf.variable_scope('main'):
        mu,pi,logp_pi,q1,q2 = mlp_actor_critic(o=o_ph,a=a_ph,**ac_kwargs)
    with tf.variable_scope('main',reuse=True):
        _,_,_,q1_pi,q2_pi = mlp_actor_critic(o=o_ph,a=pi,**ac_kwargs)
        _,pi_next,logp_pi_next,_,_ = mlp_actor_critic(o=o2_ph,a=a_ph,**ac_kwargs)
    # Target value
    with tf.variable_scope('target'):
        _,_,_,q1_targ,q2_targ = mlp_actor_critic(o=o2_ph,a=pi_next,**ac_kwargs)
        
    # Get variables
    main_vars,q_vars,pi_vars,target_vars = \
        get_vars('main'),get_vars('main/q'),get_vars('main/pi'),get_vars('target')
    
    model = {'o_ph':o_ph,'a_ph':a_ph,'o2_ph':o2_ph,'r_ph':r_ph,'d_ph':d_ph,
             'mu':mu,'pi':pi,'logp_pi':logp_pi,'q1':q1,'q2':q2,
             'q1_pi':q1_pi,'q2_pi':q2_pi,
             'pi_next':pi_next,'logp_pi_next':logp_pi_next,
             'q1_targ':q1_targ,'q2_targ':q2_targ,
             'main_vars':main_vars,'q_vars':q_vars,'pi_vars':pi_vars,'target_vars':target_vars}
        
    return model,sess

def create_sac_graph(model,lr=1e-3,gamma=0.98,alpha_q=0.1,alpha_pi=0.1,polyak=0.995,epsilon=1e-2):
    """
    SAC Computational Graph
    """
    # Double Q-learning
    min_q_pi = tf.minimum(model['q1_pi'],model['q2_pi'])
    min_q_targ = tf.minimum(model['q1_targ'],model['q2_targ'])
    
    # Entropy-regularized Bellman backup
    q_backup = tf.stop_gradient(
        model['r_ph'] + 
        gamma*(1-model['d_ph'])*(min_q_targ - alpha_q*model['logp_pi_next'])
    )
    
    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha_pi*model['logp_pi'] - min_q_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - model['q1'])**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - model['q2'])**2)
    value_loss = q1_loss + q2_loss
    
    # Policy train op
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr,epsilon=epsilon)
    train_pi_op = pi_optimizer.minimize(pi_loss,var_list=model['pi_vars'])
    
    # Value train op 
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr,epsilon=epsilon)
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss,var_list=model['q_vars'])
        
    # Polyak averaging for target variables
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in 
                                      zip(model['main_vars'], model['target_vars'])]
                                )
    
    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, model['q1'], model['q2'], model['logp_pi'],
                train_pi_op, train_value_op, target_update]
    
    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in 
                                zip(model['main_vars'], model['target_vars'])]
                          )

    return step_ops,target_init
    
def get_action(model,sess,o,deterministic=False):
    act_op = model['mu'] if deterministic else model['pi']
    return sess.run(act_op, feed_dict={model['o_ph']:o.reshape(1,-1)})[0]


def save_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,
                               VERBOSE=True,IGNORE_BUFFERS=False):
    """
    Save SAC model weights and replay buffers
    """
    
    # SAC model
    tf_vars = R.model['target_vars'] + R.model['main_vars']
    data2save,var_names,var_vals = dict(),[],[]
    for v_idx,tf_var in enumerate(tf_vars):
        var_name,var_val = tf_var.name,R.sess.run(tf_var)
        var_names.append(var_name)
        var_vals.append(var_val)
        data2save[var_name] = var_val
        if VERBOSE:
            print ("[%02d]  var_name:[%s]  var_shape:%s"%
                (v_idx,var_name,var_val.shape,)) 
            
    # Buffers
    if IGNORE_BUFFERS is False:
        names_long,vals_long = replay_buffer_long.get()
        names_short,vals_short = replay_buffer_short.get()
        for name,val in zip(names_long,vals_long):
            data2save[name+'_long'] = val
        for name,val in zip(names_short,vals_short):
            data2save[name+'_short'] = val
    
    # Create folder if not exist
    dir_name = os.path.dirname(npz_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print ("[%s] created."%(dir_name))
        
    # Save npz
    np.savez(npz_path,**data2save)
    print ("[%s] saved."%(npz_path))
            
            
def restore_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,
                                  VERBOSE=True,IGNORE_BUFFERS=False):
    """
    Restore SAC model weights and replay buffers
    """
    
    # Load npz
    l = np.load(npz_path)
    print ("[%s] loaded."%(npz_path))
    
    # Get values of SAC model  
    tf_vars = R.model['target_vars'] + R.model['main_vars']
    var_vals = []
    for tf_var in tf_vars:
        var_vals.append(l[tf_var.name])   
        
    # Assign weights of SAC model
    R.set_weights(var_vals)
    
    # Restore buffers
    if IGNORE_BUFFERS is False:
        buffer_names,_ = replay_buffer_long.get()
        a = []
        for buffer_name in buffer_names:
            a.append(l[buffer_name+'_long'])
        replay_buffer_long.restore(a)
        a = []
        for buffer_name in buffer_names:
            a.append(l[buffer_name+'_short'])
        replay_buffer_short.restore(a)
    
    