# -*- coding: utf-8 -*-
# from RL_brain import DQNPrioritizedReplay
# from google_rl import DQNPrioritizedReplay
from drqn_rl import DQNPrioritizedReplay
import tensorflow as tf
import numpy as np
from drl.envs.envs_register import ENVS
from drl.envs.vfh_planner import vfh_plan, forward_move, get_close_vw
import rospy
from drl.utils.logx import log, Logger, EpochLogger, ExitClass
from drl.utils.run_utils import setup_logger_kwargs
from collections import deque
import math
import copy
import os

#按需分配
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#定量分配
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
#config = tf.ConfigProto(gpu_options=gpu_options)
#sess = tf.Session(config=config)

is_train = False
is_restore_model =True

ENV_TYPE = 'gazebo_stage_pc_env'
#ENV_TYPE = 'gazebo_all_image_with_global_planner'
#ENV_CFGS = ['train_v2_mix.yaml'] #复杂静态
#ENV_CFGS = ['multi.yaml'] #多机器人 scan_img: 4.launch
ENV_CFGS = ['normal_difficulty_env.yaml','normal_difficulty_env.yaml','hard_difficulty_env.yaml','expert_difficulty_env.yaml']
MAP_CHECK = False
#ENV_TYPE = 'stage_all_image'
#ENV_CFGS = 'stage_test.yaml'

MEMORY_SIZE = 50000
BATCH_SIZE = 1024
NET_TYPE = 'drqn' # net_small
EXP_NAME = 'pc_drqn_8random'
Learning_rate = 0.001
E_greedy_increment = 0.000025
frames_per_timestep = 1
steps_per_plan = 3
lstm_size = 512

if is_train:
    env = ENVS[ENV_TYPE](ENV_CFGS)
    #env.view_ranges = [[0.5, 3, 0.5, 3]]
    env.view_ranges = [[0.5, 1, 0.5, 1], [1, 2, 1, 2],[1,3,1,3],[2,3,2,3]]
else:
    env = ENVS[ENV_TYPE](['normal_difficulty_env.yaml'])
    env.view_ranges = [[2, 3, 2, 3]]

env.image_batch = frames_per_timestep
env.steps_per_planning = steps_per_plan


RL_prio = DQNPrioritizedReplay(
        n_actions=len(env.discrete_actions), image_size=[env.image_size[0], env.image_size[1], env.image_batch],
        state_dim=env.state_dim,
        learning_rate=Learning_rate, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
        e_greedy_increment=E_greedy_increment, restore_model=is_restore_model, prioritized=False, output_graph=True,
        exp_name=EXP_NAME, net_type=NET_TYPE, is_train=is_train, n_steps=env.steps_per_planning
    )
#sess.run(tf.global_variables_initializer())
USE_VFH = False

info_logger_kwargs = setup_logger_kwargs(EXP_NAME)
info_logger = EpochLogger(**info_logger_kwargs)
train_info = {}
train_info['model'] = {}
train_info['model']['type'] = NET_TYPE
train_info['model']['batch_size'] = BATCH_SIZE
train_info['model']['memory_size'] = MEMORY_SIZE
train_info['model']['Learning_rate'] = Learning_rate
train_info['model']['e_greedy_increment'] = E_greedy_increment
train_info['env'] = {}
train_info['env']['type'] = ENV_TYPE
train_info['env']['cfg_yamls'] = ENV_CFGS
train_info['env']['actions'] = env.discrete_actions
train_info['env']['image_size'] = [env.image_size[0], env.image_size[1], env.image_batch]
train_info['env']['state_dim'] = env.state_dim
info_logger.save_config(train_info)

def train(RL):
    total_steps = 0
    robot_total = env.robot_total
    epochs = 1000000
    steps_per_epoch = 1500
    episode_max_len = 200
    for epoch in range(epochs):
        env.epoch = epoch
        #RL_prio.set_ep()
        observations = env.reset(target_dist = 0.5,draw_goal=True, map_check=MAP_CHECK)
        observe_horizons = []
        observe_states = []
        c_state_in = np.zeros((1, lstm_size), dtype=np.float32)
        h_state_in = np.zeros((1, lstm_size), dtype=np.float32)
        for i_ in range(robot_total):
            tmp_deque_h = deque(maxlen=env.steps_per_planning)
            tmp_deque_c = deque(maxlen=env.steps_per_planning)
            observe_horizons.append(tmp_deque_h)
            observe_states.append(tmp_deque_c)
            for _ in range(env.steps_per_planning):
                observe_horizons[i_].append(copy.deepcopy(observations[1][i_]))
                observe_states[i_].append(copy.deepcopy(observations[0][i_]))
                # print "horizons[i_].shape", observe_horizons[i_]
                # print "observe_states[i_].shape", observe_states[i_]
        ep_lens = [0.0] * robot_total
        ep_returns = [0.0] * robot_total
        finish_msgs = []
        for step in range(steps_per_epoch):
            actions = []
            for o_i in range(robot_total):
                states_input = np.array(observe_states[o_i])
                horizons_input = np.array(observe_horizons[o_i])
                action = RL.choose_action(observation=[states_input, horizons_input], actions=env.discrete_actions,
                                          c_state_in=c_state_in, h_state_in=h_state_in,
                                          is_test=False, use_vfh=USE_VFH)
                actions.append(action)
            observations_, rewards, dones = env.step_discrete(actions)
            for i in range(robot_total):
                if env.dones[i] == -3:
                    finish_msgs.append("Robot {0} finish : Goal Out".format(i))
                    env.dones[i] = -100
                if env.dones[i] < 0 and ep_lens[i] <= steps_per_plan:
                    finish_msgs.append("Robot {0} finish : Too Short".format(i))
                    env.dones[i] = -100
                if env.dones[i] == -1:
                    finish_msgs.append("Robot {0} finish : Collision".format(i))
                if env.dones[i] == -2:
                    finish_msgs.append("Robot {0} finish : Arrive".format(i))
                if env.dones[i] != -100:
                    RL.store_transition([observations[0][i], observations[1][i]],
                                        actions[i], rewards[i],
                                        [observations_[0][i], observations_[1][i]], env.dones[i] < 0)
                    ep_returns[i] += rewards[i]
                    ep_lens[i] += 1
                    if env.dones[i] < 0:
                        env.dones[i] = -100
                if ep_lens[i] > episode_max_len:
                    env.dones[i] = -100
            observations = observations_
            for i_r in range(robot_total):
                observe_horizons[i_r].append(copy.deepcopy(observations[1][i_r]))
                observe_states[i_r].append(copy.deepcopy(observations[0][i_r]))
            for msg in finish_msgs:
                rospy.logwarn(msg)
            if env.dones.count(-100) == robot_total:
                finish_msgs = []
                print "----------------"
                print "epoch: ", epoch
                print "ep_returns: ", ep_returns
                print "ep_lens", ep_lens
                env.robots_control([[0, 0]] * robot_total)
                RL.logger.store(EpRet=ep_returns[0], EpLen=ep_lens[0])
                # RL.logger.store(EpRet2=ep_returns[1], EpLen2=ep_lens[1])
                # RL.logger.store(EpRet3=ep_returns[2], EpLen3=ep_lens[2])
                # RL.logger.store(EpRet4=ep_returns[3], EpLen4=ep_lens[3])
                total_steps += np.array(ep_lens).sum()
                if total_steps > BATCH_SIZE and np.array(ep_lens).sum() > robot_total:
                    # for _ in range(int(np.array(ep_lens).sum())):
                    for i in range(50):
                        # abs_errors, loss = RL.learn()
                        # RL.logger.store(AbsErrors=abs_errors, Loss=loss)
                        loss = RL.learn()
                        RL.logger.store(Loss=loss)
                ep_lens = [0.0] * robot_total
                ep_returns = [0.0] * robot_total
                observations = env.reset(target_dist=0.5, draw_goal=True, map_check=MAP_CHECK)
                for i_b in range(robot_total):
                    for _ in range(env.steps_per_planning):
                        observe_horizons[i_b].append(copy.deepcopy(observations[1][i_b]))
                        observe_states[i_b].append(copy.deepcopy(observations[0][i_b]))
        RL.save_train_model()
        if total_steps > BATCH_SIZE:
            RL.logger.log_tabular('Epoch', epoch)
            RL.logger.log_tabular('Step', total_steps)
            RL.logger.log_tabular('EpLen', average_only=True)
            # RL.logger.log_tabular('EpLen2', average_only=True)
            # RL.logger.log_tabular('EpLen3', average_only=True)
            # RL.logger.log_tabular('EpLen4', average_only=True)
            RL.logger.log_tabular('EpRet', with_min_and_max=True)
            # RL.logger.log_tabular('EpRet2', with_min_and_max=True)
            # RL.logger.log_tabular('EpRet3', with_min_and_max=True)
            # RL.logger.log_tabular('EpRet4', with_min_and_max=True)
            RL.logger.log_tabular('Loss', with_min_and_max=True)
            #RL.logger.log_tabular('AbsErrors', with_min_and_max=True)
            RL.logger.log_tabular('epsilon', RL_prio.epsilon)
            RL.logger.dump_tabular()
            #reach_rate = test(RL,10)
            #if epoch > 30:
                #RL.set_ep(reach_rate)

r_risk = 0.8
r_safe = 1.0
v_max = 0.4

def hybrid_controller_test(RL,observations,o_i):
    min_obs_dis = observations[2][o_i]

    if min_obs_dis <= r_risk:
        print('safe policy')
        cur_v = observations[5][o_i][0]
        print('cur_v:',cur_v)
        if cur_v > v_max:
            print('safe policy stop')
            action = 3
        else:
            # TODO 对激光看到的距离进行放缩
            action = RL.choose_action(observation=[states_input, horizons_input], actions=env.discrete_actions,
                                          c_state_in=c_state_in, h_state_in=h_state_in,
                                          is_test=False, use_vfh=USE_VFH)
            real_v = env.discrete_actions[action]
            if real_v[0] > v_max:
                print('safe policy clip')
                real_v[0] = v_max
            '''
            if min_dis_xy[1] > 0:
                print('turn right [0.12, -0.2]')
                real_v[1] -= 0.2
            else:
                print('turn left + 0.2]')
                real_v[1] += 0.2
            '''
            action = env.discrete_actions.index(real_v)
        return action

    else:
        return RL.choose_action([observations[0][o_i], observations[1][o_i],observations[4][o_i]], env.discrete_actions, is_test=True,use_vfh=USE_VFH)


# TODO
def test(RL,test_replay_):
    robot_total = env.robot_total
    dt = env.control_hz
    env.set_colis_dist(0.3)
    test_replay = test_replay_
    collision = 0.0
    stuck = 0.0
    reach = 0.0
    av_reward = 0.0
    av_r_obs = 0.0
    #r_obs=control_hz*(1/min_dist[1] + 1/min_dist[2] + ...1/min_dist[total_step]
    step_per_collision = 0.0
    step_per_reach = 0.0
    av_vmax = 0.0
    av_v = 0.0
    av_trajectory_length = 0.0
    av_total_delta_v = 0.0
    av_wmax = 0.0
    av_w = 0.0
    av_total_theta = 0.0
    av_total_delta_w = 0.0

    for i in range(test_replay):
        print('i---------------')
        observations = env.reset(target_dist = 0.5,draw_goal=True, map_check=MAP_CHECK)         
        observe_horizons = []
        observe_states = []
        c_state_in = np.zeros((robot_total, 1, lstm_size), dtype=np.float32)
        h_state_in = np.zeros((robot_total, 1, lstm_size), dtype=np.float32)
        for i_ in range(robot_total):
            tmp_deque_h = deque(maxlen=env.steps_per_planning)
            tmp_deque_c = deque(maxlen=env.steps_per_planning)
            observe_horizons.append(tmp_deque_h)
            observe_states.append(tmp_deque_c)
            for _ in range(env.steps_per_planning):
                observe_horizons[i_].append(copy.deepcopy(observations[1][i_]))
                observe_states[i_].append(copy.deepcopy(observations[0][i_]))
        steps = 0
        r_obs = [0.0] * robot_total
        is_end = [0] * robot_total
        ep_lens = [0.0] * robot_total
        ep_returns = [0.0] * robot_total
        last_actions = [3] * robot_total
        vmax = [0.0] * robot_total
        trajectory_length = [0.0] * robot_total
        total_delta_v = [0.0] * robot_total
        wmax = [0.0] * robot_total
        total_theta = [0.0] * robot_total
        total_delta_w = [0.0] * robot_total

        while True:
            steps += 1
            actions = []
            for o_i in range(robot_total):
                states_input = np.array(observe_states[o_i])
                horizons_input = np.array(observe_horizons[o_i])
                action = RL.choose_action(observation=[states_input, horizons_input], actions=env.discrete_actions,
                                          c_state_in=c_state_in[o_i], h_state_in=h_state_in[o_i],
                                          is_test=False, use_vfh=USE_VFH)
                #action = hybrid_controller_test(RL,observations,o_i)
                actions.append(action)
                v = env.discrete_actions[action][0]
                w = env.discrete_actions[action][1]
                if v > vmax[o_i]:
                    vmax[o_i] = v
                if math.fabs(w) > wmax[o_i]:
                    wmax[o_i] = math.fabs(w)
                trajectory_length[o_i] += dt * v
                total_theta[o_i] += dt * math.fabs(w)
                total_delta_v[o_i] += math.fabs(v - env.discrete_actions[last_actions[o_i]][0])
                total_delta_w[o_i] += math.fabs(w - env.discrete_actions[last_actions[o_i]][1])

            observations_, rewards, dones = env.step_discrete(actions)

            for o_i in range(robot_total):
                if is_end[o_i] == 0:
                    ep_returns[o_i] += rewards[o_i]
                    ep_lens[o_i] += 1
                    r_obs[o_i] += dt * 1.0 / observations_[2][o_i]

                    if env.dones[o_i] < 0 or steps > 200:
                        is_end[o_i] = 1
                        av_reward += ep_returns[o_i]
                        av_vmax += vmax[o_i]
                        av_wmax += wmax[o_i]
                        if env.dones[o_i] == -1:
                            collision += 1
                            step_per_collision += ep_lens[o_i]
                        elif env.dones[o_i] == -2:
                            reach += 1
                            step_per_reach += ep_lens[o_i]
                            av_r_obs += r_obs[o_i]
                            av_trajectory_length += trajectory_length[o_i]
                            av_total_theta += total_theta[o_i]
                            av_total_delta_v += total_delta_v[o_i]
                            av_total_delta_w += total_delta_w[o_i]
                        elif steps > 200:
                            stuck += 1

            if is_end.count(1) == robot_total or steps > 200:
                env.robots_control([[0, 0]] * robot_total)
                break
            observations = observations_
            for i_r in range(robot_total):
                observe_horizons[i_r].append(copy.deepcopy(observations[1][i_r]))
                observe_states[i_r].append(copy.deepcopy(observations[0][i_r]))
            last_actions = actions
    if reach != 0:
        step_per_reach = step_per_reach / reach
        av_r_obs = av_r_obs / reach
        av_trajectory_length = av_trajectory_length / reach
        av_total_delta_v = av_total_delta_v / reach
        av_total_theta = av_total_theta / reach
        av_total_delta_w = av_total_delta_w / reach
        av_v = av_trajectory_length / (step_per_reach * dt)
        av_w = av_total_theta / (step_per_reach * dt)
    if collision != 0:
        step_per_collision = step_per_collision / collision
    av_reward = av_reward / robot_total / test_replay
    av_vmax = av_vmax / robot_total / test_replay
    av_wmax = av_wmax / robot_total / test_replay

    RL.logger.log_tabular('replay_times', test_replay)#测试次数
    RL.logger.log_tabular('reach_rate', reach / robot_total / test_replay)#到达率
    RL.logger.log_tabular('reach_av_step', step_per_reach)#所有到达的episode,的平均step（时间=step×control_hz）
    RL.logger.log_tabular('collision_rate', collision / robot_total / test_replay)#碰撞率
    RL.logger.log_tabular('collision_av_step', step_per_collision)#所有发生碰撞的episode，的平均step
    RL.logger.log_tabular('stuck_rate', stuck / robot_total / test_replay)#超时未到达的概率
    RL.logger.log_tabular('average_Reward', av_reward)#所有episode，平均reward
    RL.logger.log_tabular('average_R_obs', av_r_obs)#用于衡量行驶路线上robot和障碍物的距离，越贴着障碍物走则数值越大
    RL.logger.log_tabular('average_vmax', av_vmax)#所有episode，平均最大线速度
    RL.logger.log_tabular('average_trajectory_length', av_trajectory_length)#所有到达的episode，平均路线长
    RL.logger.log_tabular('average_v', av_v)#所有到达的episode，平均线速度
    RL.logger.log_tabular('average_total_delta_v', av_total_delta_v)#所有到达的episode，线速度变化dv的绝对值之和
    RL.logger.log_tabular('average_wmax', av_wmax)#所有episode，平均最大角速度
    RL.logger.log_tabular('average_total_theta', av_total_theta)#所有到达的episode，robot转过的角度的绝对值之和
    RL.logger.log_tabular('average_w', av_w)#所有到达的episode，平均角速度
    RL.logger.log_tabular('average_total_delta_w', av_total_delta_w)#所有到达的episode，角速度变化dw的绝对值之和
    RL.logger.dump_tabular()
    return(reach / robot_total / test_replay)


if is_train:
    #forward_move("/kejiarobot0/scan")
    train(RL_prio)
else:
    test(RL_prio, test_replay_=200)

