
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import time
from geometry_msgs.msg import PointStamped, PoseStamped, Twist

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Int32

from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import sys
import tensorflow as tf
# import tensorflow_datasets as tfds
# import rlds
# from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import tf_agents
# from tf_agents.trajectories import time_step as ts
# from IPython import display
# from collections import defaultdict
# import matplotlib.pyplot as plt
import tensorflow_hub as hub
from cv_bridge import CvBridge
import cv2
from skimage.transform import resize
from datetime import datetime
import matplotlib.pyplot as plt
import json

#fix seeds
np.random.seed(0)
tf.random.set_seed(0)




commands = {
	'coke':[
		"Pick up the coke can on the table",
		"Grab the can of Coke from the table.",
		"Retrieve the Coke can that is on the table.",
		"Take the Coke can off the table.",
		"Collect the Coke can sitting on the table.",
		"Lift the Coke can off the table."
	],
	'yellow':[
		"Pick up the yellow bottle can on the table",
		"Grab the can of yellow bottle from the table.",
		"Retrieve the yellow bottle can that is on the table.",
		"Take the yellow bottle can off the table.",
		"Collect the yellow bottle can sitting on the table.",
		"Lift the yellow bottle can off the table."
	],
	'black':[
		"Pick up the black cup can on the table",
		"Grab the can of black cup from the table.",
		"Retrieve the black cup can that is on the table.",
		"Take the black cup can off the table.",
		"Collect the black cup can sitting on the table.",
		"Lift the black cup can off the table."
	],
	'cupboard':[
		"Open the cupboard door",
		"Open the door of the cupboard.",
		"Unlock the cupboard door.",
		"Pull open the cupboard door.",
		"Swing open the cupboard door.",
		"Unlatch and open the cupboard door."
	]
}

#COMMAND TYPES
#1. pick up coke
#2. cupboard
COMMAND_TYPE = 'coke'
COMMANDS = commands[COMMAND_TYPE]
# COMMAND = "Open the cupboard door"
WHEEL_SEPARATION = 0.3153
WHEEL_DIAMETER = 0.1016


universal_sentence_encoder = hub.load('../universal')
print("INITIALIZED SENTENCE ENCODER")
command_embeddings = universal_sentence_encoder(COMMANDS)
command_embedding = np.mean(command_embeddings, axis=0)
command_embedding = np.expand_dims(command_embedding, axis=0)



joints = ['wrist_extension', 'joint_lift', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_head_pan', 'joint_head_tilt', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_finger_left', 'joint_gripper_finger_right']

joint_limits = {
	'wrist_extension':[0, 1],
	'joint_lift':[0, 0.8],
	'joint_head_pan':[-4.04, 1.73],
	'joint_head_tilt':[-1.53, 0.79],
	'joint_wrist_yaw':[-1.39, 4.42],
	'joint_wrist_pitch':[-1.57, 0.56],
	'joint_wrist_roll':[-3.14, 3.14],
}

bridge = CvBridge()

OUTPUT_DIR = '../output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR_FOR_RUN = os.path.join(OUTPUT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(OUTPUT_DIR_FOR_RUN)
pic_dir = os.path.join(OUTPUT_DIR_FOR_RUN, 'pics')
os.makedirs(pic_dir)

RT1_LOWER_LIMIT = -1.0
RT1_UPPER_LIMIT = 1.0

RT1_GRIPPER_UPPER_LIMIT = 1.0
RT1_GRIPPER_LOWER_LIMIT = -1.0
ACTUATOR_GRIPPER_UPPER_LIMIT = 1.0
ACTUATOR_GRIPPER_LOWER_LIMIT = 0.0

ACTUATOR_LIFT_LOWER_LIMIT = 0
ACTUATOR_LIFT_UPPER_LIMIT = 1.2

LIFT_HEIGHT = 1.2

RT1_ACTION_RECORD = {
	'head0':{
		'world_vector_x':[],
		'world_vector_y':[],
		'world_vector_z':[],
		'world_vector_r':[],
		'gripper_close':[],
		'roll':[],
		'pitch':[],
		'yaw':[],
		'terminate_episode':[]
	}
}

REAL_ACTION_RECORD = {
	'x':[],
	'y':[],
	'z':[],
	'r':[],
	'theta':[],
	'pitch':[],
	'yaw':[],
	'roll':[],
	'gripper_close':[],
	'is_terminal_episode':[]

}

def plot_rt1_actions():
	print('saving rt1 actions')
	fig, ax = plt.subplots(len(RT1_ACTION_RECORD['head0']), figsize=(20, 50))
	for idx, (key, value) in enumerate(RT1_ACTION_RECORD['head0'].items()):
		ax[idx].plot(value)
		for camera_key, value in RT1_ACTION_RECORD.items():
			if camera_key == 'head0':
				continue
			else:	
				ax[idx].plot(RT1_ACTION_RECORD[camera_key][key])
		ax[idx].set_title(key)
		ax[idx].legend(list(RT1_ACTION_RECORD.keys()))
	plt.savefig(os.path.join(OUTPUT_DIR_FOR_RUN, f'rt1_actions.png'))
	fig.clf()
	plt.clf()
	plt.cla()

def plot_real_actions():
	print('saving real actions')
	fig, ax = plt.subplots(len(REAL_ACTION_RECORD), figsize=(20, 50))
	for idx, (key, value) in enumerate(REAL_ACTION_RECORD.items()):
		ax[idx].plot(value)
		ax[idx].set_title(key)
	plt.savefig(os.path.join(OUTPUT_DIR_FOR_RUN, 'real_actions.png'))
	plt.clf()
	plt.cla()
	
def diff_drive_inv_kinematics(V:float,omega:float)->tuple:
	#COPIED FROM STRETCH MUJOCO REPO
	"""
	Calculate the rotational velocities of the left and right wheels for a differential drive robot."""
	R =WHEEL_DIAMETER/ 2
	L = WHEEL_SEPARATION
	if R <= 0:
		raise ValueError("Radius must be greater than zero.")
	if L <= 0:
		raise ValueError("Distance between wheels must be greater than zero.")
	
	# Calculate the rotational velocities of the wheels
	w_left = (V - (omega * L / 2)) / R
	w_right = (V + (omega * L / 2)) / R

	return (w_left, w_right)

def record_real_actions(real_action):
	REAL_ACTION_RECORD['x'].append(real_action['x'])
	REAL_ACTION_RECORD['y'].append(real_action['y'])
	REAL_ACTION_RECORD['z'].append(real_action['z'])
	REAL_ACTION_RECORD['r'].append(real_action['r'])
	REAL_ACTION_RECORD['theta'].append(real_action['theta'])
	REAL_ACTION_RECORD['pitch'].append(real_action['pitch'])
	REAL_ACTION_RECORD['yaw'].append(real_action['yaw'])
	REAL_ACTION_RECORD['roll'].append(real_action['roll'])
	REAL_ACTION_RECORD['gripper_close'].append(real_action['gripper_close'])
	REAL_ACTION_RECORD['is_terminal_episode'].append(real_action['is_terminal_episode'])

def record_rt1_actions(rt1_action, camera='head'):
	base_displacement_vector = rt1_action['base_displacement_vector']
	base_displacement_vector_x = base_displacement_vector[0]
	base_displacement_vector_y = base_displacement_vector[1]
	omega = np.arctan2(base_displacement_vector_y, base_displacement_vector_x)

	base_rotation = rt1_action['base_displacement_vertical_rotation']
	gripper_closedness = rt1_action['gripper_closedness_action']
	rotation_delta = rt1_action['rotation_delta']
	rt1_roll = rotation_delta[0]
	rt1_pitch = rotation_delta[1]
	rt1_yaw = rotation_delta[2]

	terminate_episode = rt1_action['terminate_episode']
	world_vector = rt1_action['world_vector']
	world_vector_x = world_vector[0]
	world_vector_y = world_vector[1]
	world_vector_z = world_vector[2]

	world_vector_r = np.sqrt(world_vector_x**2 + world_vector_y**2)
	world_vector_theta = np.arctan2(world_vector_y, world_vector_x) + np.pi/2
	if camera not in RT1_ACTION_RECORD.keys():
		RT1_ACTION_RECORD[camera] = {
			'world_vector_x':[],
			'world_vector_y':[],
			'world_vector_z':[],
			'world_vector_r':[],
			'gripper_close':[],
			'roll':[],
			'pitch':[],
			'yaw':[],
			'terminate_episode':[]
		}

	RT1_ACTION_RECORD[camera]['world_vector_x'].append(world_vector_x)
	RT1_ACTION_RECORD[camera]['world_vector_y'].append(world_vector_y)
	RT1_ACTION_RECORD[camera]['world_vector_z'].append(world_vector_z)
	RT1_ACTION_RECORD[camera]['world_vector_r'].append(world_vector_r)
	RT1_ACTION_RECORD[camera]['gripper_close'].append(gripper_closedness)
	RT1_ACTION_RECORD[camera]['roll'].append(rt1_roll)
	RT1_ACTION_RECORD[camera]['pitch'].append(rt1_pitch)
	RT1_ACTION_RECORD[camera]['yaw'].append(rt1_yaw)
	RT1_ACTION_RECORD[camera]['terminate_episode'].append(terminate_episode)

def mix_rt1_action_outputs_from_multiple_inputs(rt1_actions):
	number_of_rt1_actions = len(rt1_actions)
	mixed_rt1_actions = {}
	for rt1_action in rt1_actions:
		for key, val in rt1_action.items():
			if key not in mixed_rt1_actions:
				mixed_rt1_actions[key] = val
			else:
				mixed_rt1_actions[key] += val
	print(mixed_rt1_actions)
	for key in mixed_rt1_actions.keys():
		mixed_rt1_actions[key] = mixed_rt1_actions[key] / number_of_rt1_actions
	return mixed_rt1_actions


def mix_real_action_outputs_from_multiple_inputs(real_actions):
	number_of_real_actions = len(real_actions)
	head_real_action = real_actions[0]
	head2_real_action = real_actions[1]
	gripper_real_action = real_actions[2]

	mixed_real_actions = {}
	for real_action in real_actions:
		for key, val in real_action.items():
			if key not in mixed_real_actions:
				mixed_real_actions[key] = val
			else:
				mixed_real_actions[key] += val
	for key in mixed_real_actions.keys():
		mixed_real_actions[key] /= number_of_real_actions

	# we are prioritizing the grippers decision when it comes to the control of gripper
	mixed_real_actions['gripper_close'] = head_real_action['gripper_close']
	
	
	return mixed_real_actions
		

class RT1Node(Node):
	def __init__(self):
		super().__init__('rt1_controller')
		model_path = '../rt_1_x_tf_trained_for_002272480_step'
		self.head_rt1_num = 4
		self.wrist_rt1_num = 2

		self.camera_type = 'head'
		# only one of below should be true
		self.alternate_images = False
		self.alternate_every_n_steps = 5
		self.mix_multiple_inputs = True
		self.wrist_rt1_num = 2

		self.rt1_head_tfa_policies = []
		self.rt1_head_observations = []
		self.rt1_head_policy_states = []

		self.rt1_wrist_tfa_policies = []
		self.rt1_wrist_observations = []
		self.rt1_wrist_policy_states = []


		for i in range(self.head_rt1_num):
			tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
				model_path=model_path,
				load_specs_from_pbtxt=True,
				use_tf_function=True)
			self.rt1_head_tfa_policies.append(tfa_policy)
			self.rt1_head_policy_states.append(tfa_policy.get_initial_state(batch_size=1))
			self.rt1_head_observations.append(tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)))
			print(f'INITIALIZED RT1 HEAD {i}')

		if self.alternate_images or self.mix_multiple_inputs:
			for i in range(self.wrist_rt1_num):
				tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
					model_path=model_path,
					load_specs_from_pbtxt=True,
					use_tf_function=True)
				self.rt1_wrist_tfa_policies.append(tfa_policy)
				self.rt1_wrist_policy_states.append(tfa_policy.get_initial_state(batch_size=1))
				self.rt1_wrist_observations.append(tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)))
				print(f'INITIALIZED RT1 WRIST {i}')


		self.step_num = 0

		self.got_head_image = False
		self.got_wrist_image = False
		self.got_joint_states = False


		print('Connecting to action server')

		#joint trajectory client
		self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
		server_reached = self.trajectory_client.wait_for_server(timeout_sec=15.0)
		if not server_reached:
			self.get_logger().error('Unable to connect to arm action server. Timeout exceeded.')
			sys.exit()
		self.joint_subscription = self.create_subscription(JointState, '/stretch/joint_states', self.joint_states_callback, 1)

		# subscribers
		self.sound_direction_sub = self.create_subscription(Int32, "/sound_direction", self.callback_direction, 1)
		self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 1)
		self.gripper_image_sub = self.create_subscription(Image, '/gripper_camera/color/image_rect_raw', self.wrist_image_callback, 1)
		self.target_point_publisher = self.create_publisher(PointStamped, "/clicked_point", 1)
		# cmd vel publisher
		self.twist_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 1)
		
		self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')


		# states
		self.joint_state = JointState()
		self.image_state = None
		self.resized_image_state = None
		self.gripper_image_state = None
		self.resized_gripper_image_state = None

	def record_configs_for_run(self):
		data = {
			'camera_type':self.camera_type,
			'alternate_images':self.alternate_images,
			'alternate_every_n_steps':self.alternate_every_n_steps,
			'mix_multiple_inputs': self.mix_multiple_inputs,
			'head_rt1_num':self.head_rt1_num,
			'wrist_rt1_num':self.wrist_rt1_num
		}
		with open(os.path.join(OUTPUT_DIR_FOR_RUN, 'config.json'), 'w') as f:
			json.dump(data, f)


	def translate_rt1_action_to_real_actions(self, rt1_action):
		head_pan_idx = self.joint_state.name.index('joint_head_pan')
		head_tilt_idx = self.joint_state.name.index('joint_head_tilt')
		wrist_yaw_idx = self.joint_state.name.index('joint_wrist_yaw')
		wrist_pitch_idx = self.joint_state.name.index('joint_wrist_pitch')
		wrist_roll_idx = self.joint_state.name.index('joint_wrist_roll')
		gripper_left_idx = self.joint_state.name.index('joint_gripper_finger_left')
		gripper_right_idx = self.joint_state.name.index('joint_gripper_finger_right')
		lift_idx = self.joint_state.name.index('joint_lift')
		extension_idx = self.joint_state.name.index('wrist_extension')

		prev_head_pan_value = self.joint_state.position[head_pan_idx]
		prev_head_tilt_value = self.joint_state.position[head_tilt_idx]
		prev_wrist_yaw_value = self.joint_state.position[wrist_yaw_idx]
		prev_wrist_pitch_value = self.joint_state.position[wrist_pitch_idx]
		prev_wrist_roll_value = self.joint_state.position[wrist_roll_idx]
		prev_gripper_left_value = self.joint_state.position[gripper_left_idx]
		prev_gripper_right_value = self.joint_state.position[gripper_right_idx]
		prev_lift_value = self.joint_state.position[lift_idx]
		prev_extension_value = self.joint_state.position[extension_idx]
		# RT1 outputs decomposed
		base_displacement_vector = rt1_action['base_displacement_vector']
		base_displacement_vector_x = base_displacement_vector[0]
		base_displacement_vector_y = base_displacement_vector[1]
		omega = np.arctan2(base_displacement_vector_y, base_displacement_vector_x)

		rotation_delta = rt1_action['rotation_delta']
		rt1_roll = rotation_delta[0]
		rt1_pitch = rotation_delta[1]
		rt1_yaw = rotation_delta[2]


		base_rotation = rt1_action['base_displacement_vertical_rotation']
		gripper_closedness = rt1_action['gripper_closedness_action']
		rotation_delta = rt1_action['rotation_delta']
		rt1_roll = rotation_delta[0]
		rt1_pitch = rotation_delta[1]
		rt1_yaw = rotation_delta[2]


		terminate_episode = rt1_action['terminate_episode']
		is_terminal_episode = terminate_episode[0]
		world_vector = rt1_action['world_vector']
		world_vector_x = world_vector[0]
		world_vector_y = world_vector[1]
		world_vector_z = world_vector[2]

		world_vector_r = np.sqrt(world_vector_x**2 + world_vector_y**2)
		world_vector_theta = np.arctan2(world_vector_y, world_vector_x) + np.pi/2

		return {
			'x':world_vector_x,
			'y':world_vector_y,
			'extension': world_vector_x+0.19,
			'z':float((world_vector_z- RT1_LOWER_LIMIT) / (RT1_UPPER_LIMIT - RT1_LOWER_LIMIT))*(ACTUATOR_LIFT_UPPER_LIMIT - ACTUATOR_LIFT_LOWER_LIMIT) + ACTUATOR_LIFT_LOWER_LIMIT + 0.05,
			'r':float(world_vector_r + 0.2),
			'theta':world_vector_theta,
			'pitch': rt1_pitch,
			'yaw': rt1_yaw,
			'roll': rt1_roll,
			'gripper_close': float((gripper_closedness*-1 - RT1_GRIPPER_LOWER_LIMIT) / (RT1_GRIPPER_UPPER_LIMIT - RT1_GRIPPER_LOWER_LIMIT)*(ACTUATOR_GRIPPER_UPPER_LIMIT - ACTUATOR_GRIPPER_LOWER_LIMIT) + ACTUATOR_GRIPPER_LOWER_LIMIT),
			# 'gripper_close': float(gripper_closedness),
			'is_terminal_episode':is_terminal_episode
		}
	
	def average_real_action_outputs_from_multiple_inputs(self, real_actions):
		number_of_real_actions = len(real_actions)
		number_of_head_rt1 = self.head_rt1_num
		head_real_actions = real_actions[:number_of_head_rt1]

		average_real_action = {}
		for real_action in real_actions:
			for key, val in real_action.items():
				if key not in average_real_action:
					average_real_action[key] = val
				else:
					average_real_action[key] += val
		for key in average_real_action.keys():
			average_real_action[key] /= number_of_real_actions

		average_head_real_actions = {}
		for head_real_actions in head_real_actions:
			for key, val in head_real_actions.items():
				if key not in average_head_real_actions:
					average_head_real_actions[key] = val
				else:
					average_head_real_actions[key] += val
		for key in average_head_real_actions.keys():
			average_head_real_actions[key] /= number_of_head_rt1

		# we are prioritizing the grippers decision when it comes to the control of gripper
		average_real_action['gripper_close'] = average_head_real_actions['gripper_close']
		# we are also prioritizing the head y coordinate over the grippers
		average_real_action['y'] = average_head_real_actions['y']
		
		
		return average_real_action
		

	def main(self):
		self.record_configs_for_run()
		while True:
			try:
				self.got_head_image = False
				self.got_wrist_image = False
				self.got_joint_states = False

				while not self.got_head_image or not self.got_wrist_image:
					print('----------------------------------')
					print('state received')
					print('got_head_image', self.got_head_image)
					print('got_wrist_image', self.got_wrist_image)
					rclpy.spin_once(self)
					print('----------------------------------')
				
				self.step_num += 1
				# print('image shape', self.image_state.shape)
				if self.mix_multiple_inputs:
					images = [self.resized_image_state, self.resized_gripper_image_state]
					image_names = ['head', 'gripper']
					action_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
					real_actions = []
					rt1_actions = []

					for image, name in zip(images, image_names):
						# save picture 
						cv2.imwrite(os.path.join(pic_dir, f'{action_time_str}_{name}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
						if name == 'head':
							for i in range(self.head_rt1_num):
								self.rt1_head_observations[i]['image'] = image
								self.rt1_head_observations[i]['natural_language_embedding'] = command_embedding
								tfa_time_step = ts.transition(self.rt1_head_observations[i], reward=np.zeros((), dtype=np.float32))
								policy_step = self.rt1_head_tfa_policies[i].action(tfa_time_step, self.rt1_head_policy_states[i])
								rt1_action = policy_step.action
								self.rt1_head_policy_states[i] = policy_step.state
								print('head rt1 action')
								print(rt1_action)
								record_rt1_actions(rt1_action, f'head{i}')
								real_action = self.translate_rt1_action_to_real_actions(rt1_action)
								real_actions.append(real_action)
								rt1_actions.append(rt1_action)

						elif name == 'gripper':
							for i in range(self.wrist_rt1_num):
								self.rt1_wrist_observations[i]['image'] = image
								self.rt1_wrist_observations[i]['natural_language_embedding'] = command_embedding
								tfa_time_step = ts.transition(self.rt1_wrist_observations[i], reward=np.zeros((), dtype=np.float32))
								policy_step = self.rt1_wrist_tfa_policies[i].action(tfa_time_step, self.rt1_wrist_policy_states[i])
								rt1_action = policy_step.action
								self.rt1_wrist_policy_states[i] = policy_step.state
								record_rt1_actions(rt1_action, f'gripper{i}')
								real_action = self.translate_rt1_action_to_real_actions(rt1_action)
								real_actions.append(real_action)
								rt1_actions.append(rt1_action)
						
						else:
							raise ValueError('Unknown image name')
						
						
					# mixed_rt1_action = mix_rt1_action_outputs_from_multiple_inputs(rt1_actions)
					# record_rt1_actions(mixed_rt1_action)
					real_action = self.average_real_action_outputs_from_multiple_inputs(real_actions)
					record_real_actions(real_action)

				else:
					if self.alternate_images and self.step_num % self.alternate_every_n_steps == 0:
						print('alternating image input')
						print('before', self.camera_type)
						self.camera_type = 'wrist' if self.camera_type == 'head' else 'head'
						print('after', self.camera_type)
						print('----------------------------------')
					rt1_image =  self.resized_gripper_image_state if self.camera_type == 'wrist' else self.resized_image_state
					self.rt1_observation['image'] = rt1_image
					action_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
					# save picture 
					cv2.imwrite(os.path.join(pic_dir, f'{action_time_str}.png'), cv2.cvtColor(rt1_image, cv2.COLOR_RGB2BGR))

					tfa_time_step = ts.transition(self.rt1_observation, reward=np.zeros((), dtype=np.float32))

					policy_step = self.tfa_policy.action(tfa_time_step, self.rt1_policy_state)
					rt1_action = policy_step.action
					record_rt1_actions(rt1_action)
					self.rt1_policy_state = policy_step.state
					real_action = self.translate_rt1_action_to_real_actions(rt1_action)
					record_real_actions(real_action)

				is_terminal_episode = real_action['is_terminal_episode']
				if is_terminal_episode == 1.0:
					plot_real_actions()
					plot_rt1_actions()
					
					self.get_logger().info('Terminal Episode. Shutting Down Node...')
					self.destroy_node()
					rclpy.shutdown()
					break
				# rotation = real_action['theta']

				# publishing rotation
				twist = Twist() 
				# twist.angular.z = rotation
				twist.linear.x = float(real_action['y'])
				self.twist_pub.publish(twist)
				print('---------------------------------------------------------------')
				print('RT1 ACTION')
				print(rt1_action)
				print('---------------------------------------------------------------')


				
				# duration0 = Duration(seconds=0.0)
				duration1 = Duration(seconds=10.0)

				head_pan_idx = self.joint_state.name.index('joint_head_pan')
				head_tilt_idx = self.joint_state.name.index('joint_head_tilt')
				wrist_yaw_idx = self.joint_state.name.index('joint_wrist_yaw')
				wrist_pitch_idx = self.joint_state.name.index('joint_wrist_pitch')
				wrist_roll_idx = self.joint_state.name.index('joint_wrist_roll')
				gripper_left_idx = self.joint_state.name.index('joint_gripper_finger_left')
				gripper_right_idx = self.joint_state.name.index('joint_gripper_finger_right')
				lift_idx = self.joint_state.name.index('joint_lift')
				extension_idx = self.joint_state.name.index('wrist_extension')

				head_pan_value = self.joint_state.position[head_pan_idx]
				head_tilt_value = self.joint_state.position[head_tilt_idx]
				wrist_yaw_value = self.joint_state.position[wrist_yaw_idx]
				wrist_pitch_value = self.joint_state.position[wrist_pitch_idx]
				wrist_roll_value = self.joint_state.position[wrist_roll_idx]
				gripper_left_value = self.joint_state.position[gripper_left_idx]
				gripper_right_value = self.joint_state.position[gripper_right_idx]
				lift_value = self.joint_state.position[lift_idx]
				extension_value = self.joint_state.position[extension_idx]

				lift_height_remain = LIFT_HEIGHT - lift_value
				angle = np.arctan2(real_action['extension'] + 0.5, lift_height_remain)
				print("ANGLE", angle)
				 

				head_pan_pos = -np.pi/2
				head_tilt_pos = -np.pi/2 + angle
				print('HEAD_TILT_POS', head_tilt_pos)
				wrist_yaw_pos = np.pi/2
				wrist_pitch_pos = 0.0

				# FIRST POINT
				# point0 = JointTrajectoryPoint()
				prev_positions = [
						lift_value, 
						extension_value, 
						wrist_yaw_value, 
						wrist_pitch_value,
						wrist_roll_value,
						head_pan_value, 
						head_tilt_value,
						gripper_left_value,
					]
				

				# point0.positions = prev_positions
				
				# point0.velocities = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
				# point0.accelerations = [1.0, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0]
				# point0.time_from_start = duration0.to_msg()

				# SECOND POINT
				point1 = JointTrajectoryPoint()
				positions = [
						# map lift value to 0.2 to 1.0
						float(real_action['z']),
						float(real_action['extension']), 
						float(real_action['yaw']),
						# wrist_yaw_pos,
						float(real_action['pitch']),
						# wrist_pitch_pos,
						float(real_action['roll']),
						head_pan_pos,
						head_tilt_pos,
						float(real_action['gripper_close'])
						# -1.0
						]
				
				# if positions[0] < prev_positions[0]:
				# 	# making sure it does not go down compared to previous
				# 	positions[0] = prev_positions[0]
				
				# if positions[0] < 0.58:
				# 	positions[0] = lift_value
				joint_names = [
					'joint_lift', 
					'wrist_extension', 
					'joint_wrist_yaw', 
					'joint_wrist_pitch',
					'joint_wrist_roll',
					'joint_head_pan',
					'joint_head_tilt',
					'joint_gripper_finger_left',
					]
				
				
				print('previous')
				print(' '.join(joint_names))
				print(' '.join(map(str, prev_positions)))
				print('current')
				print(' '.join(joint_names))
				print(' '.join(map(str, positions)))
					
				point1.positions = positions
				
				point1.time_from_start = duration1.to_msg()
				trajectory_goal = FollowJointTrajectory.Goal()
				trajectory_goal.trajectory.joint_names = joint_names
				trajectory_goal.trajectory.points = [point1]
				trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
				trajectory_goal.trajectory.header.frame_id = 'base_link'
				print('sending trajectory goal')
				self.trajectory_client.send_goal_async(trajectory_goal)
				plot_real_actions()
				plot_rt1_actions()
				print('done sending trajectory goal')
			except KeyboardInterrupt:
				plot_rt1_actions()
				self.get_logger().info('Keyboard Interrupt. Shutting Down Node...')
				self.destroy_node()
				rclpy.shutdown()
				break

	def callback_speech(self, msg):
		pass

	def callback_direction(self, msg):
		pass

	def image_callback(self, msg):
		print('image received')
		cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		np_image = np.rot90(np.array(cv_image), k=3) 
		np_image = np_image[400:, :, :]
		resized = cv2.resize(np_image,dsize=(320, 256), interpolation=cv2.INTER_CUBIC)
		#save
		cv2.imwrite('image.png', cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
		cv2.imwrite('resized_image.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
		self.image_state = np_image
		self.resized_image_state = resized
		self.got_head_image = True

	def wrist_image_callback(self, msg):
		print('wrist image received')
		cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		np_image = np.array(cv_image)
		resized = cv2.resize(np_image,dsize=(320, 256), interpolation=cv2.INTER_CUBIC)
		#save
		cv2.imwrite('wrist_image.png', cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
		cv2.imwrite('resized_wrist_image.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
		self.gripper_image_state = np_image
		self.resized_gripper_image_state = resized
		self.got_wrist_image = True

	def joint_states_callback(self, joint_state):
		print('joint_state received')
		self.joint_state = joint_state
		self.got_joint_states = True

		print('--------------------------------------------------------------')
	
	def get_logger(self):
		return super().get_logger()
	
	

def main(args=None):
	rclpy.init(args=args) 

	node = RT1Node()
	print('node initialized')
	try:

		# wait for joint state and image to come in

		while node.image_state is None:
			print('waiting for image')
			rclpy.spin_once(node)

		while node.gripper_image_state is None:
			print('waiting for gripper image')
			rclpy.spin_once(node)
			print('spun once')

		# node.issue_multipoint_command()
		node.main()

		node.destroy_node()
		rclpy.shutdown()
	except KeyboardInterrupt:
		
		node.get_logger().info('Keyboard Interrupt. Shutting Down Node...')
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()