import numpy as np
import tensorflow as tf
from util import gpu_sess,suppress_tf_warning


def create_ars_model(odim=10,adim=2,hdims=[128],
                     actv=tf.nn.relu,out_actv=tf.nn.tanh):
    """
    Augmented Random Search Model
    """
    import tensorflow as tf
    
    def mlp(x,hdims=[256,256],actv=tf.nn.relu,out_actv=tf.nn.relu):
        ki = tf.truncated_normal_initializer(stddev=0.1)
        for hdim in hdims[:-1]:
            x = tf.layers.dense(x,units=hdim,activation=actv,kernel_initializer=ki)
        return tf.layers.dense(x,units=hdims[-1],
                               activation=out_actv,kernel_initializer=ki)
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
    o_ph = placeholder(odim)
    
    # Policy 
    with tf.variable_scope('main'):
        mu = mlp(o_ph,hdims=hdims+[adim],actv=actv,out_actv=out_actv)
    
    # Params
    main_vars = get_vars('main')
    
    model = {'o_ph':o_ph,'mu':mu,'main_vars':main_vars}
    return model, sess
    
    
def get_noises_from_weights(weights,nu=0.01):
    """
    Get noises from weights
    """
    noises = []
    for weight in weights:
        noise = nu*np.random.randn(*weight.shape)
        noises.append(noise)
    return noises