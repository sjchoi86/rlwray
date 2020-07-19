import datetime,gym,time,os,psutil,ray
import numpy as np
import tensorflow as tf
from util import gpu_sess,suppress_tf_warning,tic,toc,open_txt,write_txt
from sac import ReplayBuffer,create_sac_model,create_sac_graph,\
    save_sac_model_and_buffers,restore_sac_model_and_buffers
np.set_printoptions(precision=2)
suppress_tf_warning() # suppress warning 
gym.logger.set_level(40) # gym logger 
# from episci.environment_wrappers.tactical_action_adt_env_continuous import CustomADTEnvContinuous
from episci.agents.utils.constants import Agents

def train(expname='sac_adt_cont',n_cpu=30,n_workers=30,
          total_steps=50000,burnin_steps=10,
          evaluate_every=50,print_every=5,
          action_length=5,action_length_eval=5,
          ep_len_rollout=10*150,
          hdims=[128,128],actv=tf.nn.relu,
          red_list_train = {Agents.SPOT_4G: 0.15,Agents.SPOT_5G: 0.30,Agents.SPOT_RANDOM: 0.45,
                            Agents.EXPERT_SYSTEM_TRIAL_2: 0.6,Agents.EXPERT_SYSTEM_TRIAL_3_SCRIMMAGE_4: 0.75,
                            Agents.EXPERT_SYSTEM: 1.0},
          red_list_eval=[Agents.SPOT_RANDOM,Agents.EXPERT_SYSTEM]*5,
          num_eval=5,max_ep_len_eval=15e3,
          batch_size=2**12,update_count=500,
          lr=1e-4,epsilon=1e-5,
          gamma=0.99,alpha_q=0.05,alpha_pi=0.5,polyak=0.995,
          buffer_sz_long=1e7,buffer_sz_short=1e6,
          temp_min=1.0,temp_max=1.0,eps_greedy=0.0,
          txt_path='../log/sac_adt_cont/log_time.txt',
          npz_path_restore=''):
    """
    Train SAC model 
    """
    
    # Logger
    print (txt_path)
    f = open_txt(txt_path)
    print ("[%s] created."%(txt_path))
    time.sleep(1) # wait 
    
    # Environments
    def get_env():
        from episci.environment_wrappers.tactical_action_adt_env_continuous import CustomADTEnvContinuous
        from episci.agents.utils.constants import Agents, RewardType
        red_distribution = red_list_train
        env_config = {
            "red_distribution": red_distribution,
            "reward_type": RewardType.SHAPED
        }
        return CustomADTEnvContinuous(env_config,action_length=action_length)

    def get_eval_env():
        from episci.environment_wrappers.tactical_action_adt_env_continuous import CustomADTEnvContinuous
        from episci.agents.utils.constants import Agents, RewardType
        red_distribution = red_list_train
        env_config = {
            "red_distribution": red_distribution,
            "reward_type": RewardType.SHAPED
        }
        return CustomADTEnvContinuous(env_config,action_length=action_length_eval)
    
    # Rollout Worker
    class RolloutWorkerClass(object):
        """
        Worker without RAY (for update purposes)
        """
        def __init__(self,hdims=[256,256],actv=tf.nn.relu,
                     lr=1e-3,gamma=0.99,alpha_q=0.1,alpha_pi=0.1,polyak=0.995,epsilon=1e-2,
                     seed=1):
            self.seed = seed
            # Each worker should maintain its own environment
            import gym
            from util import suppress_tf_warning
            suppress_tf_warning() # suppress TF warnings
            gym.logger.set_level(40) 
            self.env = get_eval_env()
            odim,adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
            self.odim = odim
            self.adim = adim
            _ = self.env.reset()

            # Create SAC model and computational graph 
            self.model,self.sess = create_sac_model(
                odim=self.odim,adim=self.adim,hdims=hdims,actv=actv)
            self.step_ops,self.target_init = \
                create_sac_graph(self.model,lr=lr,gamma=gamma,alpha_q=alpha_q,alpha_pi=alpha_pi,
                                 polyak=polyak,epsilon=epsilon)

            # Initialize model 
            self.FIRST_SET_FLAG = True
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_init)

        def get_action(self,o,deterministic=False):
            act_op = self.model['mu'] if deterministic else self.model['pi']
            return self.sess.run(act_op, feed_dict={self.model['o_ph']:o.reshape(1,-1)})[0]

        def get_weights(self):
            """
            Get weights
            """
            weight_vals = self.sess.run(self.model['main_vars'])
            return weight_vals
        
        def set_weights(self,weight_vals):
            """
            Set weights without memory leakage
            """
            if self.FIRST_SET_FLAG:
                self.FIRST_SET_FLAG = False
                self.assign_placeholders = []
                self.assign_ops = []
                for w_idx,weight_tf_var in enumerate(self.model['main_vars']):
                    a = weight_tf_var
                    assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                    assign_op = a.assign(assign_placeholder)
                    self.assign_placeholders.append(assign_placeholder)
                    self.assign_ops.append(assign_op)
            for w_idx,weight_tf_var in enumerate(self.model['main_vars']):
                # Memory-leakage-free assign (hopefully)
                self.sess.run(self.assign_ops[w_idx],
                              {self.assign_placeholders[w_idx]:weight_vals[w_idx]})

    @ray.remote
    class RayRolloutWorkerClass(object):
        """
        Rollout Worker with RAY
        """
        def __init__(self,worker_id=0,hdims=[256,256],actv=tf.nn.relu,
                     ep_len_rollout=1000,max_ep_len_eval=1000):
            # Parse
            self.worker_id = worker_id
            self.ep_len_rollout = ep_len_rollout
            self.max_ep_len_eval = max_ep_len_eval
            # Each worker should maintain its own environment
            import gym
            from util import suppress_tf_warning
            suppress_tf_warning() # suppress TF warnings
            gym.logger.set_level(40) 
            self.env = get_env()
            odim,adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
            self.odim = odim
            self.adim = adim
            _ = self.env.reset()

            # Replay buffers to pass
            self.o_buffer = np.zeros((self.ep_len_rollout,self.odim))
            self.a_buffer = np.zeros((self.ep_len_rollout,self.adim))
            self.r_buffer = np.zeros((self.ep_len_rollout))
            self.o2_buffer = np.zeros((self.ep_len_rollout,self.odim))
            self.d_buffer = np.zeros((self.ep_len_rollout))

            # Create SAC model
            self.model,self.sess = create_sac_model(
                odim=self.odim,adim=self.adim,hdims=hdims,actv=actv)
            self.sess.run(tf.global_variables_initializer())
            print ("Ray Worker [%d] Ready."%(self.worker_id))

            # Flag to initialize assign operations for 'set_weights()'
            self.FIRST_SET_FLAG = True

            # Flag to initialize rollout
            self.FIRST_ROLLOUT_FLAG = True

        def get_action(self,o,deterministic=False,temperature=1.0):
            """
            Get action (if temperature is 0, it becomes deterministic)
            """
            a_mu = self.sess.run(self.model['mu'],
                                 feed_dict={self.model['o_ph']:o.reshape(1,-1)})[0]
            a_pi = self.sess.run(self.model['pi'],
                                 feed_dict={self.model['o_ph']:o.reshape(1,-1)})[0]
            if deterministic:
                a = a_mu
            else:
                a = temperature*a_pi + (1-temperature)*a_mu
            return a

        def set_weights(self,weight_vals):
            """
            Set weights without memory leakage
            """
            if self.FIRST_SET_FLAG:
                self.FIRST_SET_FLAG = False
                self.assign_placeholders = []
                self.assign_ops = []
                for w_idx,weight_tf_var in enumerate(self.model['main_vars']):
                    a = weight_tf_var
                    assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                    assign_op = a.assign(assign_placeholder)
                    self.assign_placeholders.append(assign_placeholder)
                    self.assign_ops.append(assign_op)
            for w_idx,weight_tf_var in enumerate(self.model['main_vars']):
                # Memory-leakage-free assign (hopefully)
                self.sess.run(self.assign_ops[w_idx],
                              {self.assign_placeholders[w_idx]:weight_vals[w_idx]})

        def rollout(self,temperature=1.0,eps_greedy=0.0):
            """
            Rollout
            """
            if self.FIRST_ROLLOUT_FLAG:
                self.FIRST_ROLLOUT_FLAG = False
                self.o = self.env.reset() # reset environment
            # Loop
            r_sum = 0
            for t in range(self.ep_len_rollout):
                if np.random.rand() < eps_greedy:
                    self.a = self.env.action_space.sample() # random sample 
                else:
                    self.a = self.get_action(self.o,deterministic=False,temperature=temperature)
                self.o2,self.r,self.d,_ = self.env.step(self.a)
                r_sum += self.r
                # Append
                self.o_buffer[t,:] = self.o
                self.a_buffer[t,:] = self.a
                self.r_buffer[t] = self.r
                self.o2_buffer[t,:] = self.o2
                self.d_buffer[t] = self.d
                # Save next state 
                self.o = self.o2
                if self.d: 
                    self.o = self.env.reset() # reset when done 
            r_avg = r_sum / self.ep_len_rollout
            return self.o_buffer,self.a_buffer,self.r_buffer,self.o2_buffer,self.d_buffer,r_avg

        def evaluate(self,red=None):
            """
            Evaluate
            """
            o,d,ep_ret,ep_len = self.env.reset(red=red),False,0,0
            while not(d or (ep_len == self.max_ep_len_eval)):
                a = self.get_action(o,deterministic=True)
                o,r,d,_ = self.env.step(a)
                ep_ret += r # compute return 
                ep_len += 1
            blue_health,red_health = self.env.blue_health,self.env.red_health
            eval_res = [ep_ret,ep_len,blue_health,red_health] # evaluation result 
            return eval_res
        
    # Write down important hyper params
    write_txt(f,
              "lr:[%.2e], epsilon:[%.2e], gamma:[%.4f], alpha_q:[%.4f], alpha_pi:[%.4f]"%
              (lr,epsilon,gamma,alpha_q,alpha_pi),
              ADD_NEWLINE=True,DO_PRINT=False)
    
    # Initialize Workers
    """
    ray.init(num_cpus=n_cpu,
             memory = 5*1024*1024*1024,
             object_store_memory = 10*1024*1024*1024,
             driver_object_store_memory = 5*1024*1024*1024)
    """
    ray.init(num_cpus=n_cpu)
    tf.reset_default_graph()
    R = RolloutWorkerClass(hdims=hdims,actv=actv,
                           lr=lr,gamma=gamma,alpha_q=alpha_q,alpha_pi=alpha_pi,
                           polyak=polyak,epsilon=epsilon,
                           seed=0)
    workers = [RayRolloutWorkerClass.remote(worker_id=i,hdims=hdims,actv=actv,
                                            ep_len_rollout=ep_len_rollout,
                                            max_ep_len_eval=max_ep_len_eval) 
               for i in range(n_workers)]
    print ("RAY initialized with [%d] cpus and [%d] workers."%
           (n_cpu,n_workers))
    write_txt(f,"RAY initialized with [%d] cpus and [%d] workers."%
           (n_cpu,n_workers),ADD_NEWLINE=True,DO_PRINT=False)
    
    # Replay Buffers
    replay_buffer_long = ReplayBuffer(odim=R.odim,adim=R.adim,size=int(buffer_sz_long))
    replay_buffer_short = ReplayBuffer(odim=R.odim,adim=R.adim,size=int(buffer_sz_short))
    
    
    # Restore, if necessary
    if npz_path_restore:
        # npz_path_restore = '../data/net/%s/model_and_buffers_1600.npz'%(expname)
        restore_sac_model_and_buffers(npz_path_restore,R,replay_buffer_long,replay_buffer_short,
                                      VERBOSE=False,IGNORE_BUFFERS=True)
        
    # Loop
    start_time = time.time()
    n_env_step = 0 # number of environment steps
    for t in range(int(total_steps)):
        esec = time.time()-start_time

        # Synchronize worker weights
        weights = R.get_weights()
        set_weights_list = [worker.set_weights.remote(weights) for worker in workers] 

        # Make rollout and accumulate to Buffers
        t_start = time.time()
        ops = [worker.rollout.remote(
            temperature=temp_min+(temp_max-temp_min)*np.random.rand(),
            eps_greedy=eps_greedy)
               for worker in workers]
        rollout_vals = ray.get(ops)
        r_sum = 0
        for rollout_val in rollout_vals:
            o_buffer,a_buffer,r_buffer,o2_buffer,d_buffer,r_rollout_avg = rollout_val
            r_sum += r_rollout_avg
            for i in range(ep_len_rollout):
                o,a,r,o2,d = o_buffer[i,:],a_buffer[i,:],r_buffer[i],o2_buffer[i,:],d_buffer[i]
                replay_buffer_long.store(o, a, r, o2, d) 
                replay_buffer_short.store(o, a, r, o2, d) 
                n_env_step += 1
        r_avg = r_sum / len(rollout_vals)
        sec_rollout = time.time() - t_start

        # Burnin
        if t < burnin_steps:
            continue

        # Update
        t_start = time.time()
        avg_qs = np.zeros(int(update_count))
        for c_idx in range(int(update_count)):
            batch_long = replay_buffer_long.sample_batch(batch_size//2) 
            batch_short = replay_buffer_short.sample_batch(batch_size//2) 
            feed_dict = {R.model['o_ph']: np.concatenate((batch_long['obs1'],batch_short['obs1'])),
                         R.model['o2_ph']: np.concatenate((batch_long['obs2'],batch_short['obs2'])),
                         R.model['a_ph']: np.concatenate((batch_long['acts'],batch_short['acts'])),
                         R.model['r_ph']: np.concatenate((batch_long['rews'],batch_short['rews'])),
                         R.model['d_ph']: np.concatenate((batch_long['done'],batch_short['done']))
                        }
            outs = R.sess.run(R.step_ops, feed_dict) # update 
            q1_vals,q2_vals = outs[3],outs[4]
            avg_q = 0.5*np.mean(q1_vals)+0.5*np.mean(q2_vals)
            avg_qs[c_idx] = avg_q
        sec_update = time.time() - t_start

        # Synchronize worker weights (after update)
        weights = R.get_weights()
        set_weights_list = [worker.set_weights.remote(weights) for worker in workers] 

        # Print
        if (t == burnin_steps) or (((t+1)%print_every) == 0): 
            print ("[%d/%d] n_env_step:[%.1e] rollout:[%.1f]s update:[%.1f]s r_avg:[%.4f] avg_q:[%.3f]."%
                   (t+1,total_steps,n_env_step,sec_rollout,sec_update,r_avg,np.mean(avg_qs)))
            write_txt(f,
                      "[%d/%d] n_env_step:[%.1e] rollout:[%.1f]s update:[%.1f]s r_avg:[%.4f] avg_q:[%.3f]."%
                      (t+1,total_steps,n_env_step,sec_rollout,sec_update,r_avg,np.mean(avg_qs)),
                      ADD_NEWLINE=True,DO_PRINT=False)

        # Evaluate
        if (t == burnin_steps) or (((t+1)%evaluate_every) == 0): 
            ram_percent = psutil.virtual_memory().percent # memory usage
            print ("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]."%
                   (t+1,total_steps,t/total_steps*100,
                    n_env_step,
                    time.strftime("day:[%d] %H:%M:%S", time.gmtime(time.time()-start_time)),
                    ram_percent)
                  )
            write_txt(f,
                      "[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]."%
                      (t+1,total_steps,t/total_steps*100,n_env_step,
                       time.strftime("day:[%d] %H:%M:%S", time.gmtime(time.time()-start_time)),
                       ram_percent),
                      ADD_NEWLINE=True,DO_PRINT=False)
            ops = []
            for i_idx in range(num_eval):
                worker,red = workers[i_idx],red_list_eval[i_idx]
                ops.append(worker.evaluate.remote(red=red))
            eval_vals = ray.get(ops)
            ep_ret_sum = 0
            for i_idx in range(num_eval):
                red,eval_val = red_list_eval[i_idx],eval_vals[i_idx]
                ep_ret,ep_len,blue_health,red_health = eval_val[0],eval_val[1],eval_val[2],eval_val[3]
                ep_ret_sum += ep_ret
                print (" [%d/%d] [%s] ep_ret:[%.4f] ep_len:[%d]. blue health:[%.2f] red health:[%.2f]"
                    %(i_idx,len(eval_vals),red,ep_ret,ep_len,blue_health,red_health))
                write_txt(f,
                          " [%d/%d] [%s] ep_ret:[%.4f] ep_len:[%d]. blue health:[%.2f] red health:[%.2f]"%
                          (i_idx,len(eval_vals),red,ep_ret,ep_len,blue_health,red_health),
                          ADD_NEWLINE=True,DO_PRINT=False)

            ep_ret_avg = ep_ret_sum / num_eval
            print ("[Eval. done] time:[%s] ep_ret_avg:[%.3f].\n"%
                   (time.strftime("day:[%d] %H:%M:%S", time.gmtime(time.time()-start_time)),
                    ep_ret_avg)
                  )
            write_txt(f,
                      "[Eval. done] time:[%s] ep_ret_avg:[%.3f].\n"%
                      (time.strftime("day:[%d] %H:%M:%S", time.gmtime(time.time()-start_time)),
                       ep_ret_avg),
                      ADD_NEWLINE=True,DO_PRINT=False)

            # Save current SAC model and replay buffers 
            npz_path = '../data/net/%s/model_and_buffers_%d.npz'%(expname,t+1)
            save_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,
                                       VERBOSE=False,IGNORE_BUFFERS=True)
            write_txt(f,
                      " [%s] saved."%npz_path,
                      ADD_NEWLINE=True,DO_PRINT=False)

    print ("Done.")
    
    # Close Ray
    ray.shutdown()
    
    # Save
    # Path to save the npz file 
    npz_path = '../data/net/%s/model_and_buffers_final.npz'%(expname)
    save_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,
                               VERBOSE=False,IGNORE_BUFFERS=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Some margin 