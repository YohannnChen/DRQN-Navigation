# -*- coding: utf-8 -*-
import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState, ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped,Point,PoseStamped, Pose
from std_msgs.msg import Int8
import math
import drl.utils.ros_utils as util
import random
import numpy as np
from collections import deque
from gazebo_msgs.srv import GetModelState
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy
import gz_ros
import stg_ros
import subprocess
import os.path as osp
import atexit
import yaml
from comn_pkg.msg import RobotState
from comn_pkg.srv import CheckGoalRequest,CheckGoalResponse,CheckGoal, PathPlan, MapValue, MapValueRequest, PathPlanRequest
from rospy.service import ServiceException
from visualization_msgs.msg import Marker

class GazeboStageEnv(object):
    def __init__(self, cfg_names):
        self.cfg_names = cfg_names
        self.read_yaml(cfg_names[0])
        rospy.init_node("drl")
        rospy.set_param('~visual_gazebo', False)
        self.visual_gazebo = True

        self.view_ranges = [[0.5, 1, 0.5, 1], [1, 2, 1, 2],[ 2,3,2,3]]
        #self.view_ranges = [[2,3,2,3]]
        self.circle_ranges = [1.5, 2.5]
        #self.last_state_stamp=rospy.Time(secs=0,nsecs=0)
        
        self.change_task_level_epoch = 20
        self.use_goal_out = True
        self.use_polar = True
        self.state_dim = 4
        self.laser_batch = 1
        self.image_size = (60, 60)
        self.image_batch = 1
        self.steps_per_planning = 3
        self.collision_th = 0.3
        self.pose_empty = [10, 10]
        self.map_free_value = 180
        self.path_max_length = 3.5
        self.control_hz = 0.2
        self.act_dim = 2
        self.laser_dim = 512
        self.action_space_high = np.array([0.6, 0.9])
        self.action_space_low = np.array([0, -0.9])
        self.is_normalized = True
        self.laser_max = 6.0
        self.goal_max = 6.0
        self.v_max = 1.0
        self.w_max = 2.0
        self.init_datas()
        self.visual_clear_pose = [-10, -10]
        self.visual_step = 8

        # self.discrete_actions = \
        # [[0.0, -0.6], [0.0, -0.4], [0.0, -0.2], [0.0, 0.05], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6],
        # [0.12, -0.6], [0.12, -0.4], [0.12, -0.2], [0.12, 0], [0.12, 0.2], [0.12, 0.4], [0.12, 0.6],
        # [0.24, -0.6], [0.24, -0.4], [0.24, -0.2], [0.24, 0], [0.24, 0.2], [0.24, 0.4], [0.24, 0.6],
        # [0.36, -0.6], [0.36, -0.4], [0.36, -0.2], [0.36, 0], [0.36, 0.2], [0.36, 0.4], [0.36, 0.6]]
        self.discrete_actions = \
        [[0.0, -0.9], [0.0, -0.6], [0.0, -0.3], [0.0, 0.05], [0.0, 0.3], [0.0, 0.6], [0.0, 0.9],
        [0.2, -0.9], [0.2, -0.6], [0.2, -0.3], [0.2, 0], [0.2, 0.3], [0.2, 0.6], [0.2, 0.9],
        [0.4, -0.9], [0.4, -0.6], [0.4, -0.3], [0.4, 0], [0.4, 0.3], [0.4, 0.6], [0.4, 0.9],
        [0.6, -0.9], [0.6, -0.6], [0.6, -0.3], [0.6, 0], [0.6, 0.3], [0.6, 0.6], [0.6, 0.9]]

        self.bridge = CvBridge()
        self.current_env = 0
        self.change_env_flag = False
        self.change_env_sub = rospy.Subscriber("/change_env_topic",Int8,self.change_env_callback,queue_size=1)
        self.check_bumper_sub = rospy.Subscriber("/kejia0/bumper",ContactsState,self.check_bumper_callback,queue_size=1)      # add collisions check
        self.state_subs = []
        self.vel_pubs = []
        self.goal_pubs = []
        for i in range(self.robot_total):
            self.state_subs.append(rospy.Subscriber("/" + self.robots_name[i] + "/state",
                RobotState, self.state_callback, queue_size=1))
            self.vel_pubs.append(rospy.Publisher( '/' + self.robots_name[i] + '/cmd_vel', Twist, queue_size=1))
            self.goal_pubs.append(rospy.Publisher( '/' + self.robots_name[i] + '/goal', PoseStamped, queue_size=1))
        self.actor_goal_pubs = []
        self.actor_start_pubs = []
        for i in range(self.actor_total):
            self.actor_goal_pubs.append(rospy.Publisher( '/' + self.actor_model_name + str(i) + '/SetTargetsub',
                Pose, queue_size=1))
            self.actor_start_pubs.append(rospy.Publisher( '/' + self.actor_model_name + str(i) + '/SetStartposesub',
                Pose, queue_size=1))
        self.check_goal_service_name = 'check_goal_service'
        self.get_map_value_service_name = '/map_value_service'
        self.path_plan_service_name = '/path_plan_service'
        self.marker_pub = rospy.Publisher( '/marker', Marker, queue_size=100)
        self.empty_robots()

    def _get_states(self, save_img=None):
        # states, images_last, min_dists, collisions, scans,vws, goal_outs, states_norm, \
        #     safetys, laser_raws, robot_poses, image_no_goals = self.get_robots_state()
        states, images_last, min_dists, collisions, scans,vws, goal_outs, states_norm, \
            safetys, laser_raws, robot_poses = self.get_robots_state()
        images_reshape = []
        lasers_reshape = []
        for i in range(self.robot_total):
            if save_img != None:
                cv2.imwrite(save_img + "_robot_" + str(i) + ".png", images_last[0] * 255)
            ptr = self.images_ptr[i]
            if ptr < self.image_batch:
                for j in range(ptr, self.image_batch):
                    self.images_batch[i].append(copy.deepcopy(images_last[i]))
            else:
                self.images_batch[i].append(copy.deepcopy(images_last[i]))
            self.images_ptr[i] += 1

            laser_ptr = self.lasers_ptr[i]
            if laser_ptr < self.laser_batch:
                for k in range(laser_ptr, self.laser_batch):
                    self.lasers_batch[i].append(copy.deepcopy(scans[i]))
            else:
                self.lasers_batch[i].append(copy.deepcopy(scans[i]))
            self.lasers_ptr[i] += 1

        for m in range(self.robot_total):
            images_reshape.append(np.transpose(np.array(self.images_batch[m]), (1, 2, 0)))
            lasers_reshape.append(np.transpose(np.array(self.lasers_batch[m]), (1, 0)))
        # return (np.array(states), images_reshape, min_dists, collisions, scans, np.array(vws), goal_outs,\
        #     lasers_reshape, np.array(states_norm), safetys, laser_raws, robot_poses, image_no_goals)
        return (np.array(states), images_reshape, min_dists, collisions, scans, np.array(vws), goal_outs,\
            lasers_reshape, np.array(states_norm), safetys, laser_raws, robot_poses)

    def _get_rewards(self, states, min_dists, is_collisions, goal_outs, safetys, vws,laser_raws):
        rewards = []
        dones = []
        
        distance_reward_factor = 200
        obs_reward_factor = 100
        for i in range(self.robot_total):
            time_stamp_=laser_raws[i].header.stamp
            state = states[i]
            min_dist = min_dists[i]
            vw = vws[i]
            #print 'robot ',i," dist to obs: ", min_dist
            reward = collision_reward = reach_reward = step_reward = distance_reward = rotation_reward = 0
            done = 0
            if min_dist < 1.0:
                if self.last_d_obs[i] == -1:
                    self.last_d_obs[i] = min_dist
                    collision_reward = 0
                else:
                    collision_reward = (min_dist - self.last_d_obs[i]) * obs_reward_factor
                    self.last_d_obs[i] = min_dist
            # if abs(vw[1]) >= 0.7:
            #     rotation_reward = -2 * abs(vw[1])
            # rotation_reward = -2 * abs(vw[1] - self.last_w[i])
            # self.last_w[i] = vw[1]
            # collision_reward = safetys[i] * 2
            if is_collisions[i] > 0:  #used for check collision in cameral,designed by chenyuan
                print "collision"
                done = -1
                collision_reward = -500
            #if self.last_state_stamp[i]==time_stamp_:
            #    print "scan_image die"
            #    done=-5
            elif goal_outs[i] == 1 and self.use_goal_out == True:
                done = -3
            #elif min_dist <= self.collision_th:
            #    done = -1
            #    collision_reward = -500
            # elif self.bumper_collision <= 0.001:
            #     #print "bumper collision"
            #     done = -4
            else:
                if self.use_polar:
                    d = state[0]
                else:
                    d = math.sqrt(state[0] * state[0] + state[1] * state[1])
               # print 'robot ',i," dist to goal: ", d
                if d < 0.3:
                    reach_reward = 500
                    done = -2
                else:
                    if self.last_d[i] == -1:
                        self.last_d[i] = d
                        distance_reward = 0
                    else:
                        distance_reward = (self.last_d[i] - d) * distance_reward_factor
                        self.last_d[i] = d
                    step_reward = -5
                    # distance_reward = (distance_reward_factor * 1/d)
            self.last_state_stamp[i]=time_stamp_
            reward = collision_reward + reach_reward + step_reward + distance_reward + rotation_reward
            rewards.append(reward)
            dones.append(done)
       # print 'reawrd is:', rewards
        # print 'dones',  dones
        #print "--------------"
        #print "collision_reward ", collision_reward
        #print "distance_reward", distance_reward
        #print "rotation_reward", rotation_reward
        for m in range(self.robot_total):
            if dones[m] < 0 and self.dones[m] == 0:
                self.dones[m] = dones[m]
        return (rewards, dones)

    def reset(self, target_dist = 0.0, draw_goal=False, map_check=False):
        self.visual_gazebo = rospy.get_param('~visual_gazebo')
        self.robots_control([[0, 0]] * self.robot_total)
        if self.change_env_flag:
            self.del_all_obs()
            self.read_yaml(self.cfg_names[self.current_env])
            #print('change env to: ', self.cfg_names[self.current_env])
            self.change_env_flag = False               
        self.empty_robots()
        self.empty_actor()
        self.get_avoid_areas()
        time.sleep(0.5)
        self.del_all_trajectory()   # use for visulization
        self.reset_obs()
        time.sleep(0.5)
        self.reset_robots(target_dist=target_dist, draw_goal=draw_goal, map_check=map_check)
        if self.actor_total != 0:
            time.sleep(0.5)
            self.reset_actor()
        time.sleep(1.0)
        for i in range(self.robot_total):
            self.last_d_obs[i] = -1
            self.last_d[i] = -1
            self.last_w[i] = 0
            self.dones[i] = 0
        self.reset_count += 1
        self.step_count = 0
        self.bumper_collision = 999
        return self._get_states()

    def step(self, actions):
        self.robots_control(actions)
        rospy.sleep(self.control_hz)
        self.step_count += 1
        states = self._get_states()
        if self.visual_gazebo and (self.step_count % self.visual_step == 0):
            for i in range(self.robot_total):
                self.change_model_state('arrow_pose'+str(int(self.step_count/self.visual_step))+'r'+str(i), states[11][i], 'arrow_pose')
        rw = self._get_rewards(states[0], states[2], states[3], states[6], states[9], states[5],states[10])
        return states, np.array(rw[0], dtype='float64'), np.array(rw[1])

    def step_discrete(self, actions):
        real_actions = []
        for action in actions:
            real_actions.append(self.discrete_actions[action])
        return self.step(real_actions)

    def image_trans(self, img_ros):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(img_ros, desired_encoding="passthrough")
        except CvBridgeError as e:
          print(e)
        image = cv2.resize(cv_image, self.image_size) / 255.0
        return image

    def state_callback(self, msg):
        for i in range(self.robot_total, 0, -1):
            if self.robots_name[i-1] in msg.laser_image.header.frame_id:
                self.states_last[i-1] = msg
                break

    def change_env_callback(self,msg):
        self.current_env = msg.data
        self.change_env_flag = True
        print('get new env:', self.cfg_names[self.current_env])

    def check_bumper_callback(self, msg):
        if msg.states:  
            self.bumper_collision = msg.states[0].depths[0]

    def get_avoid_areas(self):
        self.obs_avoid_areas = []
        for i in range(self.robot_total):
            if self.envs_cfg['begin_poses_type'][i] == 'fix':
                self.obs_avoid_areas.append(self.envs_cfg['begin_poses'][i][0:2] + [self.robot_radius])
            if self.envs_cfg['target_poses_type'][i] == 'fix':
                self.obs_avoid_areas.append(self.envs_cfg['target_poses'][i][0:2] + [self.robot_radius])
        self.rob_avoid_areas = []
        for i in range(len(self.envs_cfg['models'])):
            pose_range = self.envs_cfg['models_pose'][i]
            model_radius = self.envs_cfg['models_radius'][i]
            pose_type = self.envs_cfg['model_poses_type'][i]
            if pose_type == 'fix':
                self.rob_avoid_areas.append(pose_range[0:2] + [model_radius])

    # 地图辅助选择起点终点
    def reset_robots(self, target_dist=0.0, draw_goal=False, map_check=True):
        print "map_check", self.map_check
        self.init_poses = [None] * self.robot_total
        self.target_poses = [None] * self.robot_total
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.z = -1
        """ circle_range=random.uniform(self.circle_ranges[0],self.circle_ranges[1])/2 """
        for i in range(self.robot_total):
            if self.env_type == 'gazebo':
                goal_msg.header.frame_id = self.robots_name[i] + "/odom"
            elif self.env_type == 'stage':
                goal_msg.header.frame_id = self.robots_name[i] + "/world"
            self.goal_pubs[i].publish(goal_msg) # 清除上一回合目标点
            # 固定起点终点的需要先找出来，其他随机点考虑避开
            if self.envs_cfg['begin_poses_type'][i] == 'fix':
                self.init_poses[i] = self.envs_cfg['begin_poses'][i]
            if self.envs_cfg['target_poses_type'][i] == 'fix':
                self.target_poses[i] = self.envs_cfg['target_poses'][i]
            if self.envs_cfg['begin_poses_type'][i] == 'rand_angle':
                tmp_pose = self.envs_cfg['begin_poses'][i]
                self.init_poses[i] = [tmp_pose[0], tmp_pose[1], random.uniform(tmp_pose[2], tmp_pose[3])]
            if self.envs_cfg['target_poses_type'][i] == 'rand_angle':
                tmp_pose = self.envs_cfg['target_poses'][i]
                self.target_poses[i] = [tmp_pose[0], tmp_pose[1], random.uniform(tmp_pose[2], tmp_pose[3])]

        for i in range(self.robot_total):
            if self.init_poses[i] != None and self.target_poses[i] != None:
                continue # 不需要随机
            reset_init = True
            while reset_init:
                goal_fail = 0
                # 先随机起点
                if 'range' in self.envs_cfg['begin_poses_type'][i]:
                    while reset_init:
                        pose_range = self.envs_cfg['begin_poses'][i]
                        if 'multi' in self.envs_cfg['begin_poses_type'][i]: # 有多个随机区域可选
                            pose_range = pose_range[random.randint(0, len(pose_range) - 1)]
                        if len(pose_range) == 4:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                        elif len(pose_range) == 6:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                        self.view_marker(rand_pose, 1)
                        if self.map_check[i]:
                            if self.free_check_robot_map(rand_pose[0], rand_pose[1], i) == False: # 判断地图此处是否为空
                                continue
                        if self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses) and \
                            self.free_check_obj([rand_pose[0], rand_pose[1], self.robot_radius*2], self.obs_range):
                            self.init_poses[i] = rand_pose[:]
                            reset_init = False
                            print "find start"
                            break
                # 再随机终点
                if 'range' in self.envs_cfg['target_poses_type'][i]:
                    while True:
                        pose_range = self.envs_cfg['target_poses'][i]
                        self.view_marker(self.init_poses[i], 2)
                        if 'multi' in self.envs_cfg['target_poses_type'][i]: # 有多个随机区域可选
                            pose_range = pose_range[random.randint(0, len(pose_range) - 1)]
                        if 'view' in self.envs_cfg['target_poses_type'][i]: # 考虑在起点范围内随机
                            rand_pose = self.random_view(self.init_poses[i], pose_range)
                        elif len(pose_range) == 4: # 正常随机
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                        elif len(pose_range) == 6:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                            #print(rand_pose)
                        self.view_marker(rand_pose, 3)
                        map_check_res = True
                        if self.map_check[i]:
                            map_check_res = self.free_check_robot_map(rand_pose[0], rand_pose[1], i) # 判断地图此处是否为空
                        # 依次判断: 不能离起点太近，远离其他目标点， 远离其他起点， 远离已知障碍物
                        if map_check_res and (self.init_poses[i][0] - rand_pose[0]) ** 2 + (self.init_poses[i][1] - rand_pose[1]) ** 2 > target_dist ** 2 \
                            and self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses) and \
                            self.free_check_obj([rand_pose[0], rand_pose[1], self.robot_radius*2], self.obs_range):
                            if self.map_check[i] == False or \
                                self.free_check_path(self.init_poses[i][0], self.init_poses[i][1],
                                    rand_pose[0], rand_pose[1], i) == True:
                                # 起点终点之间是否可以规划出路径
                                # 成功找到终点
                                self.target_poses[i] = rand_pose[:]
                                self.view_marker(rand_pose, 4)
                                print "find end"
                                break
                        goal_fail += 1
                        # 多次路径规划失败，考虑重新随机起点
                        if goal_fail > 50:
                            reset_init = True
                            break
        for i in range(self.robot_total):
            # 根据起点初始化机器人
            self.change_model_state(self.robots_name[i], self.init_poses[i], self.robot_model_name)
            if self.visual_gazebo:
                self.change_model_state('start_pose'+str(i), self.init_poses[i], 'start_pose')
                self.change_model_state('end_pose'+str(i), self.target_poses[i], 'end_pose')
            # 向scan_img发送终点
            goal_msg.pose.position.x = self.target_poses[i][0]
            goal_msg.pose.position.y = self.target_poses[i][1]
            if draw_goal == True:
                goal_msg.pose.position.z = 0
            else:
                goal_msg.pose.position.z = 1.0
            self.goal_pubs[i].publish(goal_msg)

    def del_all_obs(self):
        for model_name in self.obs_name:
            if model_name!='room':
                gz_ros.delete_model(model_name)
        self.obs_name = []

    def del_all_trajectory(self):
        model_names = gz_ros.get_world_models() 
        print "----------: ", len(model_names), model_names
        for i in range(len(model_names)):
            if 'start_pose' in model_names[i]:
                gz_ros.delete_model(model_names[i])
                #self.change_model_state(model_names[i], [self.visual_clear_pose[0]-0.2*i, self.visual_clear_pose[1]], 'start_pose')
            if 'end_pose' in model_names[i]:
                gz_ros.delete_model(model_names[i])
                #self.change_model_state(model_names[i], [self.visual_clear_pose[0]-0.2*i, self.visual_clear_pose[1]], 'end_pose')
            if 'arrow_pose' in model_names[i]:
                gz_ros.delete_model(model_names[i])
                #self.change_model_state(model_names[i], [self.visual_clear_pose[0]-0.2*i, self.visual_clear_pose[1]], 'arrow_pose')

    def reset_obs(self):
        self.obs_range = []
        for i in range(len(self.envs_cfg['models'])):
            pose_range = self.envs_cfg['models_pose'][i]
            model_radius = self.envs_cfg['models_radius'][i]
            pose_type = self.envs_cfg['model_poses_type'][i]
            model_name = self.envs_cfg['models_name'][i]

            if pose_type == 'fix':
                if len(pose_range) == 2:
                    self.obs_range.append(pose_range + [0, model_radius])
                elif len(pose_range) == 3:
                    self.obs_range.append(pose_range + [model_radius])
            elif pose_type == 'range':
                while True:
                    if len(pose_range) == 4:
                        rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])

                    elif len(pose_range) == 6:
                        rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                    if self.free_check_obj([rand_pose[0], rand_pose[1], model_radius], self.obs_range) and \
                        self.free_check_obj([rand_pose[0], rand_pose[1], model_radius], self.obs_avoid_areas):
                        self.obs_range.append(rand_pose + [model_radius])
                        break
            self.change_model_state(model_name, self.obs_range[i][0:3], self.envs_cfg['models'][i])

    def reset_actor(self, target_dist=1.0):
        self.actor_init_poses = [None] * self.actor_total
        self.actor_target_poses = [None] * self.actor_total
        goal_msg = Pose()
        startpose_msg=Pose()
        for i in range(self.actor_total):
            # 固定起点终点的需要先找出来，其他随机点考虑避开
            if self.envs_cfg['actor_begin_poses_type'][i] == 'fix':
                self.actor_init_poses[i] = self.envs_cfg['actor_begin_poses'][i]
            if self.envs_cfg['actor_target_poses_type'][i] == 'fix':
                self.actor_target_poses[i] = self.envs_cfg['actor_target_poses'][i]
            if self.envs_cfg['actor_begin_poses_type'][i] == 'rand_angle':
                tmp_pose = self.envs_cfg['actor_begin_poses'][i]
                self.actor_init_poses[i] = [tmp_pose[0], tmp_pose[1], random.uniform(tmp_pose[2], tmp_pose[3])]
            if self.envs_cfg['actor_target_poses_type'][i] == 'rand_angle':
                tmp_pose = self.envs_cfg['actor_target_poses'][i]
                self.actor_target_poses[i] = [tmp_pose[0], tmp_pose[1], random.uniform(tmp_pose[2], tmp_pose[3])]

        for i in range(self.actor_total):
            if self.actor_init_poses[i] != None and self.actor_target_poses[i] != None:
                continue # 不需要随机
            reset_init = True
            while reset_init:
                goal_fail = 0
                # 先随机起点
                if 'range' in self.envs_cfg['actor_begin_poses_type'][i]:
                    while reset_init:
                        pose_range = self.envs_cfg['actor_begin_poses'][i]
                        if 'multi' in self.envs_cfg['actor_begin_poses_type'][i]: # 有多个随机区域可选
                            pose_range = pose_range[random.randint(0, len(pose_range) - 1)]
                        if len(pose_range) == 4:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                        elif len(pose_range) == 6:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                        if self.free_check_robot(rand_pose[0], rand_pose[1], self.actor_init_poses, self.actor_radius*2) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.actor_target_poses, self.actor_radius*2) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses, self.actor_radius + self.robot_radius) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses, self.actor_radius + self.robot_radius) and \
                            self.free_check_obj([rand_pose[0], rand_pose[1], self.actor_radius], self.obs_range):
                            self.actor_init_poses[i] = rand_pose[:]
                            reset_init = False
                            break
                # 再随机终点
                if 'range' in self.envs_cfg['actor_target_poses_type'][i]:
                    while True:
                        pose_range = self.envs_cfg['actor_target_poses'][i]
                        if 'multi' in self.envs_cfg['actor_target_poses_type'][i]: # 有多个随机区域可选
                            pose_range = pose_range[random.randint(0, len(pose_range) - 1)]
                        if len(pose_range) == 4: # 正常随机
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                        elif len(pose_range) == 6:
                            rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                        # 依次判断: 不能离起点太近，远离其他目标点， 远离其他起点， 远离已知障碍物
                        if (self.actor_init_poses[i][0] - rand_pose[0]) ** 2 + (self.actor_init_poses[i][1] - rand_pose[1]) ** 2 > target_dist ** 2 \
                            and self.free_check_robot(rand_pose[0], rand_pose[1], self.actor_init_poses, self.actor_radius*2) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.actor_target_poses, self.actor_radius*2) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.init_poses, self.actor_radius + self.robot_radius) and \
                            self.free_check_robot(rand_pose[0], rand_pose[1], self.target_poses, self.actor_radius + self.robot_radius) and \
                            self.free_check_obj([rand_pose[0], rand_pose[1], self.actor_radius], self.obs_range):
                            self.actor_target_poses[i] = rand_pose[:]
                            break
                        goal_fail += 1
                        # 多次路径规划失败，考虑重新随机起点
                        if goal_fail > 50:
                            reset_init = True
                            break
        for i in range(self.actor_total):
            # 根据起点初始化actor
            actor_name = self.actor_model_name + str(i)
            self.change_model_state(actor_name, self.actor_init_poses[i], actor_name)
            # 发送起点(actor在起点终点循环)
            startpose_msg.position.x=self.actor_init_poses[i][0]
            startpose_msg.position.y=self.actor_init_poses[i][1]
            startpose_msg.orientation.w = 1
            self.actor_start_pubs[i].publish(startpose_msg)
            # 发送终点
            goal_msg.position.x = self.actor_target_poses[i][0]
            goal_msg.position.y = self.actor_target_poses[i][1]
            goal_msg.orientation.w = 1
            self.actor_goal_pubs[i].publish(goal_msg)

    def get_model_state(self, i):
        state = self.states_last[i]
        image = self.image_trans(state.laser_image)
        #image_no_goal = state.image_no_goal
        scan = state.laser
        laser_raw = state.laser_raw
        goal_rpy = util.q_to_rpy([state.pose.orientation.x, state.pose.orientation.y,\
            state.pose.orientation.z, state.pose.orientation.w])
        if self.use_polar:
            radius = math.sqrt(state.pose.position.x**2 + state.pose.position.y**2)
            if math.fabs(state.pose.position.x) < 0.00001:
                if state.pose.position.y > 0:
                    theta = math.pi / 2
                if state.pose.position.y < 0:
                    theta = -math.pi / 2
            else:
                theta = math.atan(state.pose.position.y / state.pose.position.x**2)
            goal_pose = [radius, theta]
        else:
            goal_pose = [state.pose.position.x, state.pose.position.x]
        robot_pose = state.pose_world
        vw = [state.velocity.linear.x, state.velocity.angular.z]
        min_dist = state.min_dist.point.z
        is_collision = state.collision
        goal_out = state.goal_out
        safety = state.safety
        #return goal_pose + vw, image, min_dist
        if self.state_dim == 4:
            return goal_pose + vw, image, min_dist, is_collision, scan, vw, goal_out, safety, laser_raw, robot_pose #, image_no_goal
        if self.state_dim == 2:
            return goal_pose, image, min_dist, is_collision, scan,vw, goal_out, safety, laser_raw, robot_pose  #, image_no_goal

    def get_robots_state(self):
        states = []
        images = []
        min_dists = []
        collisions = []
        scans = []
        vws = []
        goal_outs = []
        states_norm = []
        safetys = []
        laser_raws = []
        robot_poses = []
        #image_no_goals = []
        for i in range(self.robot_total):
            # state, image, min_dist, is_collision, scan,vw, goal_out, safety,\
            #     laser_raw, robot_pose, image_no_goal = self.get_model_state(i)
            state, image, min_dist, is_collision, scan,vw, goal_out, safety,\
                laser_raw, robot_pose = self.get_model_state(i)
            state_norm = []
            vw_norm = []
            if self.is_normalized == True:
                state_norm.append(state[0] / self.goal_max + 0.5)
                state_norm.append(state[1] / self.goal_max + 0.5)
                vw_norm.append(vw[0] / self.v_max)
                vw_norm.append(vw[1] / self.w_max + 0.5)
                if self.state_dim == 4:
                    state_norm.append(vw_norm[0])
                    state_norm.append(vw_norm[1])
                scan = np.array(scan) / self.laser_max
                vws.append(vw_norm)
            else:
                vws.append(vw)
            states_norm.append(state_norm)
            states.append(state)
            images.append(image)
            #image_no_goals.append(image_no_goal)
            min_dists.append(min_dist)
            collisions.append(is_collision)
            scans.append(scan)
            goal_outs.append(goal_out)
            safetys.append(safety)
            laser_raws.append(laser_raw)
            robot_poses.append(robot_pose)
        return states, images, min_dists, collisions, scans,vws, goal_outs, states_norm,\
            safetys, laser_raws, robot_poses #, image_no_goals

    def _random_pose(self, x, y, sita):
        return [random.uniform(x[0],x[1]), random.uniform(y[0],y[1]), random.uniform(sita[0],sita[1])]

    def free_check_robot(self, x, y, robot_poses, d=None):
        if d == None:
            d = self.robot_radius*2
        for pose in robot_poses:
            if pose == None:
                continue
            test_d = math.sqrt((x-pose[0])*(x-pose[0]) + (y-pose[1])*(y-pose[1]))
            if test_d <= d:
                return False
        return True

    def random_view(self, init_pose, pose_range):
        task_i = int(self.epoch / self.change_task_level_epoch)
        if task_i > len(self.view_ranges)-1:
            task_i = len(self.view_ranges)-1
        task_view = self.view_ranges[task_i]
        rand_pose = None
        while True:
            rand_pose = self._random_pose([init_pose[0]-task_view[1], init_pose[0]+task_view[1]],
                            [init_pose[1]-task_view[3], init_pose[1]+task_view[3]],
                            [-3.14, 3.14])
            if rand_pose[0] >= init_pose[0]-task_view[0] and rand_pose[0] <= init_pose[0]+task_view[0] and \
                rand_pose[1] >= init_pose[1]-task_view[2] and rand_pose[1] <= init_pose[1]+task_view[2]:
                continue
            if rand_pose[0] >= pose_range[0] and rand_pose[0] <= pose_range[1] and \
                rand_pose[1] >= pose_range[2] and rand_pose[1] <= pose_range[3]:
                break
        return rand_pose
        
    def free_check_robot_view(self,x,y,robot_index):
        try:
            rospy.wait_for_service(self.check_goal_service_name)
            check_goal_srv = rospy.ServiceProxy(self.check_goal_service_name,CheckGoal)
            request = CheckGoalRequest()
            pose = Point()
            pose.x = x
            pose.y = y
            # if self.epoch<self.change_task_level_epoch:
            #     pose.z = 1
            # else:
            pose.z = 2
            request.robot_index = robot_index
            request.pose = pose
            res = check_goal_srv(request)
            return True,res.result
        except ServiceException, e:
            print e.message
            print "check goal service error"
            return False,False
    
    def free_check_robot_map(self,x,y,robot_index):
        try:
            rospy.wait_for_service(self.get_map_value_service_name)
            map_value_srv = rospy.ServiceProxy(self.get_map_value_service_name, MapValue)
            request = MapValueRequest()
            ps = PoseStamped()
            if self.env_type == 'gazebo':
                ps.header.frame_id = self.robots_name[robot_index] + '/odom'
            elif self.env_type == 'stage':
                ps.header.frame_id = self.robots_name[robot_index] + '/world'
            ps.header.stamp = rospy.Time.now()
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1
            request.pose = ps
            res = map_value_srv(request)
            if res.type == "true" and res.value < self.map_free_value:
                return True
            else:
                return False
        except ServiceException, e:
            print e.message
            print "map value service error"
            return False

    def free_check_path(self, x1, y1, x2, y2,robot_index):
        try:
            rospy.wait_for_service(self.path_plan_service_name)
            path_plan_srv = rospy.ServiceProxy(self.path_plan_service_name, PathPlan)
            request = PathPlanRequest()
            ps = PoseStamped()
            if self.env_type == 'gazebo':
                ps.header.frame_id = self.robots_name[robot_index] + '/odom'
            elif self.env_type == 'stage':
                ps.header.frame_id = self.robots_name[robot_index] + '/world'
            ps.header.stamp = rospy.Time.now()
            ps.pose.position.x = x1
            ps.pose.position.y = y1
            ps.pose.orientation.w = 1
            request.start = copy.deepcopy(ps)
            ps.pose.position.x = x2
            ps.pose.position.y = y2
            request.target = ps
            res = path_plan_srv(request)
            print "res.length ", res.length
            if res.success == True and res.length < self.path_max_length:
                return True
            else:
                return False
        except ServiceException, e:
            print e.message
            print "path plan service error"
            return False

    def free_check_obj(self, target_pose, obj_poses):
        for pose in obj_poses:
            if pose[-1] == 0.0:
                continue
            d = target_pose[-1] + pose[-1]
            test_d = math.sqrt((target_pose[0]-pose[0])**2 + (target_pose[1]-pose[1])**2)
            if test_d <= d:
                return False
        return True

    def random_robots_pose(self, pose_ranges):
        robot_poses = []
        for i in range(self.robot_total):
            pose_range = random.choice(pose_ranges)
            while True:
                if len(pose_range) == 4:
                    rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], [-3.14, 3.14])
                elif len(pose_range) == 6:
                    rand_pose = self._random_pose(pose_range[:2], pose_range[2:4], pose_range[4:6])
                if self.free_check_robot(rand_pose[0], rand_pose[1], robot_poses):
                    robot_poses.append(rand_pose[:])
                    break
        return robot_poses[:]

    def robots_control(self, actions):
        vel = Twist()
        for i in range(self.robot_total):
            if self.dones[i] == 0:
                vel.linear.x = actions[i][0]
                vel.angular.z = actions[i][1]
            else:
                vel.linear.x = 0
                vel.angular.z = 0
            self.vel_pubs[i].publish(vel)

    def render(self):
        pass

    def view_marker(self, pose, tp):
        mk = Marker()
        if self.env_type == 'gazebo':
            mk.header.frame_id = self.robots_name[0] + '/odom'
        elif self.env_type == 'stage':
            mk.header.frame_id = self.robots_name[0] + '/world'
        mk.header.stamp = rospy.Time.now()
        mk.ns = "my"
        mk.id = tp
        mk.type = 2
        mk.action = 0
        mk.pose.position.x = pose[0]
        mk.pose.position.y = pose[1]
        mk.pose.orientation.w = 1
        mk.scale.x = 0.5
        mk.scale.y = 0.5
        mk.scale.z = 0.5
        if tp == 1:
            mk.color.a = 1.0
            mk.color.r = 0.5
            mk.color.g = 0.0
            mk.color.b = 0.0
        elif tp == 2:
            mk.color.a = 1.0
            mk.color.r = 1.0
            mk.color.g = 0.0
            mk.color.b = 0.0
        elif tp == 3:
            mk.color.a = 1.0
            mk.color.r = 0.0
            mk.color.g = 0.0
            mk.color.b = 0.5
        elif tp == 4:
            mk.color.a = 1.0
            mk.color.r = 0.0
            mk.color.g = 0.0
            mk.color.b = 1.0
        self.marker_pub.publish(mk)

    def get_robot_name(self, i):
        return self.robot_model_name + 'x' * ((i)//10) + str((i)%10)

    def empty_robots(self):
        x = self.pose_empty[0]
        y = self.pose_empty[1]
        for i in range(self.robot_total):
            self.change_model_state(self.robots_name[i], [x, y], self.robot_model_name)
            x += 3 * self.robot_radius
    def empty_actor(self):
        x=-10
        y=-10
        for i in range(self.actor_total):
            self.change_model_state(self.actor_model_name + str(i), [x, y], self.actor_model_name)
            x += 3 * 0.5
    def change_model_state(self, model_name, state, model_sdf=None):
        if self.env_type == 'gazebo':
            if model_name in gz_ros.get_world_models():
                gz_ros.set_model_state(model_name, state)
            else:
                gz_ros.spawn_model(model_name, gz_ros.get_model_sdf(model_sdf), state, model_name)
        elif self.env_type == 'stage':
            stg_ros.set_model_state(model_name, state)

    def set_fps(self, fps):
        self.control_hz = 1.0 / fps

    def set_img_size(self, img_size):
        self.image_size = img_size
        self.init()

    def set_colis_dist(self, dist):
        self.collision_th = dist

    def read_yaml(self, yaml_file):
        self.actor_total = 0
        self.env_type = 'gazebo'
        pkg_path = gz_ros.get_pkg_path('gz_pkg')
        final_file = osp.abspath(osp.join(pkg_path, '../', 'drl', 'envs', 'cfg', yaml_file))
        print "final_file", final_file
        with open(final_file, 'r') as f:
            self.envs_cfg = yaml.load(f)
        self.robot_radius = self.envs_cfg['robot_radius']
        self.robot_model_name = self.envs_cfg['robot_model_name']
        self.robot_total = self.envs_cfg['robot_total']
        if self.envs_cfg.has_key('map_check'):
            self.map_check=self.envs_cfg['map_check']
        else:
            #self.map_check=[False,False,False,False,False,False,False,False,False,False,False,False,False]
            self.map_check=[False]*self.robot_total
        if self.envs_cfg.has_key('empty_pose'):
            self.pose_empty = self.envs_cfg['empty_pose']
        if self.envs_cfg.has_key('actor_total'):
            self.actor_total = self.envs_cfg['actor_total']
            self.actor_radius = self.envs_cfg['actor_radius']
            self.actor_model_name = self.envs_cfg['actor_model_name']
        if self.envs_cfg.has_key('env_type'):
            self.env_type = self.envs_cfg['env_type']
        self.last_state_stamp=[rospy.Time(secs=0,nsecs=0)]*self.robot_total
    def init_datas(self):
        self.reset_count = 0
        self.step_count = 0 
        self.epoch = 0
        self.last_d = []
        self.last_d_obs = []
        self.last_w = []
        self.images_batch = []
        self.images_ptr = []
        self.lasers_batch = []
        self.lasers_ptr = []
        self.states_last = []
        self.robots_name = []
        self.obs_name = []
        self.dones = []
        self.obstacles_ranges = []
        for i in range(self.robot_total):
            self.robots_name.append(self.get_robot_name(i))
            tmp_laser = deque(maxlen=self.laser_batch)
            tmp_image = deque(maxlen=self.image_batch)
            self.images_batch.append(tmp_image)
            self.lasers_batch.append(tmp_laser)
            self.states_last.append(None)
            self.images_ptr.append(0)
            self.lasers_ptr.append(0)
            self.last_d.append(-1)
            self.last_d_obs.append(-1)
            self.last_w.append(0)
            self.dones.append(0)
            self.obstacles_ranges.append([])

if __name__ == '__main__':
    rospy.init_node("test_env", anonymous=True)
    env = GazeboEnv()
    rospy.sleep(0.5)
    # env.reset()
