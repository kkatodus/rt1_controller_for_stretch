
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import time
from geometry_msgs.msg import PointStamped, Twist

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
import numpy as np
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import tf_agents
import tensorflow_hub as hub
from cv_bridge import CvBridge
import cv2
from skimage.transform import resize
from datetime import datetime
import matplotlib.pyplot as plt
import json
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image as PILImage
import logging


#fix seeds
np.random.seed(0)
tf.random.set_seed(0)
# these are commands that are embedded to be fed into the RT1s
# the embedded commands can be averaged or different embeddings can be fed into different RT1s
commands = {
	'coke':[
		"Pick up the coke can on the table",
		"Grab the can of Coke from the table.",
		"Get the Coke can that is on the table.",
		"Take the Coke can off the table.",
		"Collect the Coke can sitting on the table.",
		"Lift the Coke can off the table.",
		"Pick up the can of coke on the table",
		"Lift the can of coke off the table",
		"Retrieve the Coke can from the table.",
		"Snatch the can of Coke from the table.",
		"Fetch the Coke can placed on the table.",
		"Remove the Coke can from the table.",
		"Gather the can of Coke from the table.",
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

# command type determines which set of embeddings we will use for the RT1s
COMMAND_TYPE = 'coke'
COMMANDS = commands[COMMAND_TYPE]

WHEEL_SEPARATION = 0.3153
WHEEL_DIAMETER = 0.1016
# this is the threshold for the cosine similarity between the gripper image and the sample images to decide if we should grasp the object
GRIPPER_GRASP_THRESHOLD = 0.75

# joint names that are understood by the stretch robot
joints = ['wrist_extension', 'joint_lift', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_head_pan', 'joint_head_tilt', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_finger_left', 'joint_gripper_finger_right']

# this is something we need to convert the image message coming through ros2 to numpy array
bridge = CvBridge()

# definition of output directories and the logger
OUTPUT_DIR = '../output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR_FOR_RUN = os.path.join(OUTPUT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(OUTPUT_DIR_FOR_RUN)
pic_dir = os.path.join(OUTPUT_DIR_FOR_RUN, 'pics')
os.makedirs(pic_dir)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(OUTPUT_DIR_FOR_RUN, 'run.log'), level=logging.INFO)
logger.info(f'Starting logging for run {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

# RT1 generally outputs a float between -1 and 1
RT1_LOWER_LIMIT = -1.0
RT1_UPPER_LIMIT = 1.0

# Gripper limits of the RT1 action output
RT1_GRIPPER_UPPER_LIMIT_CLOSED = 1.0
RT1_GRIPPER_LOWER_LIMIT_OPEN = -1.0
# Gripper limits of the real action output
ACTUATOR_GRIPPER_UPPER_LIMIT_OPEN = 1.0
ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED = 0.0

ACTUATOR_LIFT_LOWER_LIMIT = 0
ACTUATOR_LIFT_UPPER_LIMIT = 1.2

LIFT_HEIGHT = 1.2

# a dictionary to record all the actions of the RT1s
RT1_ACTION_RECORD = {
}

# a dictionary to record the actions that are actually sent to the robot
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
	'is_terminal_episode':[],
	'gripper_image_similarity':[]
}

def plot_rt1_actions():
	"""
	Plotting the rt1 the output of the multiple RT1s and
	"""
	print('saving rt1 actions')
	_, ax = plt.subplots(len(RT1_ACTION_RECORD['head0']), figsize=(20, 50))
	for idx, (key, value) in enumerate(RT1_ACTION_RECORD['head0'].items()):
		ax[idx].plot(value)
		for size, (camera_key, value) in enumerate(RT1_ACTION_RECORD.items()):
			if camera_key == 'head0':
				continue
			else:	
				ax[idx].plot(RT1_ACTION_RECORD[camera_key][key], marker='o', ms=size*1.1, alpha=0.5)
		ax[idx].set_title(key)
		ax[idx].legend(list(RT1_ACTION_RECORD.keys()))
	plt.savefig(os.path.join(OUTPUT_DIR_FOR_RUN, f'rt1_actions.png'))
	plt.clf()
	plt.cla()
	plt.close()

def plot_real_actions():
	"""
	Real actions are actions that are actually sent to the robot.aka after the mapping from the rt1 action to real actions
	"""
	print('saving real actions')
	fig, ax = plt.subplots(len(list(REAL_ACTION_RECORD.keys())), figsize=(20, 50))
	for idx, (key, value) in enumerate(REAL_ACTION_RECORD.items()):
		ax[idx].plot(value)
		ax[idx].set_title(key)
	plt.savefig(os.path.join(OUTPUT_DIR_FOR_RUN, 'real_actions.png'))
	fig.clf()
	plt.clf()
	plt.cla()
	plt.close()

def cos_sim(vec1, vec2):
	"""
	:param vec1: vector 1
	:param vec2: vector 2
	:return: cosine similarity between vec1 and vec2
	"""
	return (vec1@vec2)/(torch.norm(vec1)*torch.norm(vec2))

def record_real_actions(real_action):
	"""
	:param real_action: a dictionary of the real action that is sent to the robot

	A function to record the real actions that are sent to the robot
	"""
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
	"""
	:param rt1_action: a dictionary of the rt1 action that is sent to the robot
	:param camera: the camera that the rt1 action is coming from

	A function to record the rt1 actions coming from each RT1
	"""
	base_rotation = rt1_action['base_displacement_vertical_rotation']
	gripper_closedness = rt1_action['gripper_closedness_action'][0]
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
			'base_rotation':[],
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
	RT1_ACTION_RECORD[camera]['base_rotation'].append(base_rotation)
	RT1_ACTION_RECORD[camera]['terminate_episode'].append(terminate_episode)

class RT1Node(Node):
	def __init__(self):
		super().__init__('rt1_controller')
		self.step_num = 0
		self.grasped_at_step = None

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
		# subscriber to get the joint states of the robot
		self.joint_subscription = self.create_subscription(JointState, '/stretch/joint_states', self.joint_states_callback, 1)

		# subscriber to get the images from the cameras
		self.gripper_image_sub = self.create_subscription(Image, '/gripper_camera/color/image_rect_raw', self.wrist_image_callback, 1)
		self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 1)

		# publisher to move the base of the robot
		self.twist_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 1)
		
		# states
		self.joint_state = JointState()
		self.image_state = None
		self.resized_image_state = None
		self.gripper_image_state = None
		self.resized_gripper_image_state = None

		# initializing the universal sentence encoder
		self.universal_sentence_encoder = hub.load('../universal')
		print("INITIALIZED SENTENCE ENCODER")
		self.command_embeddings = self.universal_sentence_encoder(COMMANDS)
		average_command_embedding = np.mean(self.command_embeddings, axis=0)
		self.average_command_embedding = np.expand_dims(average_command_embedding, axis=0)
		self.diversify_command_embeddings = True

		# initializing the image encoder
		self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
		self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
		# getting the sample image embeddings for the gripper
		gripper_sample_images_dir = '../grabbing_samples'
		sample_image_filenames = os.listdir(gripper_sample_images_dir)
		paths = [os.path.join(gripper_sample_images_dir, filename) for filename in sample_image_filenames]
		images = [PILImage.open(path) for path in paths]
		inputs = self.image_processor(images=images, return_tensors="pt")
		outputs = self.image_encoder(**inputs)
		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)
		self.average_sample_images_embedding = torch.mean(torch.mean(last_hidden_states, dim=1), dim=0)
		print("INITIALIZED IMAGE ENCODER")

		# initializing the RT1s, you can specify how many you want to create for the head and the wrist
		model_path = '../rt_1_x_tf_trained_for_002272480_step'
		self.head_rt1_num = 7
		self.wrist_rt1_num = 0
		# if more than this number of RT1 thinks that we should grasp the object, the grasping command is triggered
		self.rt1_gripper_number_threshold = 1
		self.has_grabbed = False
		self.release_command = False

		self.rt1_head_tfa_policies = []
		self.rt1_head_observations = []
		self.rt1_head_policy_states = []

		self.rt1_wrist_tfa_policies = []
		self.rt1_wrist_observations = []
		self.rt1_wrist_policy_states = []


		for i in range(self.head_rt1_num):
			np.random.seed(i)
			tf.random.set_seed(i)

			tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
				model_path=model_path,
				load_specs_from_pbtxt=True,
				use_tf_function=True)
			self.rt1_head_tfa_policies.append(tfa_policy)
			self.rt1_head_policy_states.append(tfa_policy.get_initial_state(batch_size=1))
			self.rt1_head_observations.append(tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)))
			print(f'INITIALIZED RT1 HEAD {i}')

		for i in range(self.wrist_rt1_num):
			np.random.seed(i)
			tf.random.set_seed(i)
			tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
				model_path=model_path,
				load_specs_from_pbtxt=True,
				use_tf_function=True)
			self.rt1_wrist_tfa_policies.append(tfa_policy)
			self.rt1_wrist_policy_states.append(tfa_policy.get_initial_state(batch_size=1))
			self.rt1_wrist_observations.append(tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)))
			print(f'INITIALIZED RT1 WRIST {i}')
		
		# after initializing all the RT1s, we wait for the user to confirm that we can continue with the task
		continue_task = False
		while not continue_task:
			answer = input('All RT1s are initialized. Should we continue with the task?(yes/no)')
			if answer == 'yes':
				continue_task = True
			time.sleep(2)
	def record_configs_for_run(self):
		"""
		A function to record the configurations of the run
		"""
		data = {
			'mixing_multiple_inputs':'Now we are always mixing multiple inputs',
			'head_rt1_num':self.head_rt1_num,
			'wrist_rt1_num':self.wrist_rt1_num,
			'diversify_command_embeddings':self.diversify_command_embeddings,
			'gripper_threshold':self.rt1_gripper_number_threshold
		}
		with open(os.path.join(OUTPUT_DIR_FOR_RUN, 'config.json'), 'w') as f:
			json.dump(data, f)


	def translate_rt1_action_to_real_actions(self, rt1_action):
		"""
		:param rt1_action: a dictionary of the rt1 action that is sent to the robot
		:return: a dictionary of the real action that is sent to the robot

		This is a key function that translates the RT1 action to the real action that is sent to the robot. We do some linear mapping of the outputs to convert the RT1 actions to the real actions to be used by stretch
		"""
		gripper_closedness = rt1_action['gripper_closedness_action'][0]
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
			'pitch': 0.0,
			'yaw': 0.0,
			'roll': 0.0,
			'gripper_close': ACTUATOR_GRIPPER_UPPER_LIMIT_OPEN-((gripper_closedness-RT1_GRIPPER_LOWER_LIMIT_OPEN)/(RT1_GRIPPER_UPPER_LIMIT_CLOSED-RT1_GRIPPER_LOWER_LIMIT_OPEN))*(ACTUATOR_GRIPPER_UPPER_LIMIT_OPEN-ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED),
			# 'gripper_close':float(gripper_closedness)*-1,
			# 'gripper_close': 1.0,
			'is_terminal_episode':is_terminal_episode
		}
	
	def get_image_embedding(self, image):
		"""
		:param image: a numpy array of the image
		:return: the embedding of the image

		This uses the Vision Transformer to get the embedding of the image
		"""
		inputs = self.image_processor(image, return_tensors='pt')
		outputs = self.image_encoder(**inputs)
		last_hidden_states = outputs.last_hidden_state
		return torch.mean(last_hidden_states, dim=1)

	def average_real_action_outputs_from_multiple_inputs(self, real_actions):
		"""
		:param real_actions: a list of dictionaries of the real actions that are sent to the robot
		:return: a dictionary of the average real action that is sent to the robot
		
		As we are using multiple instances of RT1 each producting their own RT1 action that is converted to the real action, we are averaging the real actions to get the final real action that is sent to the robot
		"""
		lift_idx = self.joint_state.name.index('joint_lift')
		extension_idx = self.joint_state.name.index('wrist_extension')

		prev_lift_value = self.joint_state.position[lift_idx]
		prev_extension_value = self.joint_state.position[extension_idx]

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
		for head_real_action in head_real_actions:
			for key, val in head_real_action.items():
				if key not in average_head_real_actions:
					average_head_real_actions[key] = val
				else:
					average_head_real_actions[key] += val
		for key in average_head_real_actions.keys():
			average_head_real_actions[key] /= number_of_head_rt1

		# counting the number of RT1s that are grabbing the object
		number_of_rt1s_grabbing = 0
		for idx, head_real_action in enumerate(head_real_actions):
			if head_real_action['gripper_close'] == ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED:
				print('Head', idx, 'Wants to grab the object')
				logger.info(f'Head {idx} wants to grab the object')
				number_of_rt1s_grabbing += 1
		
		# if more than the threshold number of RT1s are grabbing the object, we are grabbing the object
		more_than_threshold_grabbing = number_of_rt1s_grabbing >= self.rt1_gripper_number_threshold
		self.has_grabbed = more_than_threshold_grabbing

		average_real_action['gripper_close'] = ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED if more_than_threshold_grabbing else average_head_real_actions['gripper_close']

		if more_than_threshold_grabbing:
			#we want to only influence the gripper value in case we are closing it
			if not self.grasped_at_step:
				self.grasped_at_step = self.step_num
			logger.info("Grasping due as RT1s think we should grasp it")
			print('We are grabbing the object!')
			average_real_action['z'] = prev_lift_value
			average_real_action['extension'] = prev_extension_value
			average_real_action['x'] = prev_extension_value
		else:
			# we are also prioritizing the head y coordinate over the grippers
			average_real_action['y'] = average_head_real_actions['y']
			# we are also prioritizing the head z coordinate over the grippers
			average_real_action['z'] = average_head_real_actions['z']
			average_real_action['x'] = average_head_real_actions['x']		
		
		return average_real_action
		

	def carry_object(self):
		"""
		This is a command that is being sent to the robot once the target object was grabbed
		"""
		print('sending carrry object command')
		logger.info('carrying object')
		duration1 = Duration(seconds=10.0)

		point1 = JointTrajectoryPoint()
		positions = [
				# map lift value to 0.2 to 1.0
				float(1.0),
				float(0.3), 
				float(0.0),
				# wrist_yaw_pos,
				float(0.0),
				# wrist_pitch_pos,
				float(0.0),
				-np.pi/2,
				0.0,
				ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED
				# -1.0
				]
		
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
					
		point1.positions = positions
		
		point1.time_from_start = duration1.to_msg()
		trajectory_goal = FollowJointTrajectory.Goal()
		trajectory_goal.trajectory.joint_names = joint_names
		trajectory_goal.trajectory.points = [point1]
		trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
		trajectory_goal.trajectory.header.frame_id = 'base_link'
		print('sending trajectory goal')
		self.trajectory_client.send_goal_async(trajectory_goal)

	def move_cam_and_arm_to_init_pos(self):
		"""
		This is a command that is being sent to the robot in the first step to move the wrist and the head to the initial position
		"""
		logger.info('Moving wrist to init pos')
		duration1 = Duration(seconds=10.0)

		point1 = JointTrajectoryPoint()
		positions = [
				# map lift value to 0.2 to 1.0
				float(0.8),
				float(0.0), 
				float(0.0),
				# wrist_yaw_pos,
				float(0.0),
				# wrist_pitch_pos,
				float(0.0),
				-np.pi/2,
				float(-0.52),
				ACTUATOR_GRIPPER_UPPER_LIMIT_OPEN
				# -1.0
				]
		
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
					
		point1.positions = positions
		
		point1.time_from_start = duration1.to_msg()
		trajectory_goal = FollowJointTrajectory.Goal()
		trajectory_goal.trajectory.joint_names = joint_names
		trajectory_goal.trajectory.points = [point1]
		trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
		trajectory_goal.trajectory.header.frame_id = 'base_link'
		print('sending trajectory goal')
		self.trajectory_client.send_goal_async(trajectory_goal)

	def release_object(self):
		"""
		This is a command that is being sent to the robot to release the object
		"""

		logger.info('Releasing object')
		duration1 = Duration(seconds=10.0)

		point1 = JointTrajectoryPoint()
		positions = [
				ACTUATOR_GRIPPER_UPPER_LIMIT_OPEN
				]
		
		joint_names = [
					'joint_gripper_finger_left',
					]
					
		point1.positions = positions
		
		point1.time_from_start = duration1.to_msg()
		trajectory_goal = FollowJointTrajectory.Goal()
		trajectory_goal.trajectory.joint_names = joint_names
		trajectory_goal.trajectory.points = [point1]
		trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
		trajectory_goal.trajectory.header.frame_id = 'base_link'
		print('sending trajectory goal')
		self.trajectory_client.send_goal_async(trajectory_goal)
		self.release_command = False
		logger.info('Release completed')

	def main(self):
		"""
		The main function that is running the RT1 controller
		"""
		self.record_configs_for_run()
		while True:
			try:
					
				self.got_head_image = False
				self.got_wrist_image = False
				self.got_joint_states = False

				# waiting until we have all the images and states we need 
				while not self.got_head_image or not self.got_wrist_image or not self.got_joint_states:
					rclpy.spin_once(self)

				# when its the initial step, we are moving the arm and the head camera to the initial position
				if self.step_num == 0:
					self.move_cam_and_arm_to_init_pos()
					self.step_num += 1
					continue
				
				self.step_num += 1
				logger.info(f'Step num:{self.step_num}  Grasped at: Timestep {self.grasped_at_step}')
				images = [self.resized_image_state, self.resized_gripper_image_state]
				# getting the embedding of the gripper image
				gripper_image_embedding = self.get_image_embedding(self.resized_gripper_image_state)
				gripper_image_similarity = cos_sim(gripper_image_embedding[0], self.average_sample_images_embedding)
				# deciding if we need to grasp the object based on the cosine similarity of the gripper image with the sample images
				need_to_grasp = gripper_image_similarity > GRIPPER_GRASP_THRESHOLD
				REAL_ACTION_RECORD['gripper_image_similarity'].append(gripper_image_similarity.detach().numpy())
				logger.info(f'Gripper image similarity:{gripper_image_similarity}')

				if gripper_image_similarity > GRIPPER_GRASP_THRESHOLD:
					if not self.grasped_at_step:
						self.grasped_at_step = self.step_num
						logger.info(f'Grasping at timestep {self.grasped_at_step} due to consine sim')
					print('Gripper Image Similarity:', gripper_image_similarity)
					print('Gripper Image is similar to the sample images. We are grabbing the object')
				if self.grasped_at_step:
					# if we have grasped the object in the past step, we can decide whether we want to release the object or if we want to carry it
					if self.step_num > self.grasped_at_step+1:
						release = input('We are carrying the object, do you want to release it?(yes/no)')
						if release == 'yes':
							self.grasped_at_step = None
							self.release_object()
						else:
							self.carry_object()
						continue

				image_names = ['head', 'gripper']
				action_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
				real_actions = []
				rt1_actions = []

				for image, name in zip(images, image_names):
					# save picture 
					cv2.imwrite(os.path.join(pic_dir, f'{action_time_str}_{name}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
					if name == 'head':
						# iterating through the head RT1s
						for i in range(self.head_rt1_num):
							self.rt1_head_observations[i]['image'] = image
							if self.diversify_command_embeddings:
								# we are giving each rt1 a slightly different embedding for diversity
								self.rt1_head_observations[i]['natural_language_embedding'] = self.command_embeddings[i:i+1]
							else:
								self.rt1_head_observations[i]['natural_language_embedding'] = self.average_command_embedding

							tfa_time_step = ts.transition(self.rt1_head_observations[i], reward=np.zeros((), dtype=np.float32))
							policy_step = self.rt1_head_tfa_policies[i].action(tfa_time_step, self.rt1_head_policy_states[i])
							rt1_action = policy_step.action
							self.rt1_head_policy_states[i] = policy_step.state
							record_rt1_actions(rt1_action, f'head{i}')
							real_action = self.translate_rt1_action_to_real_actions(rt1_action)
							real_actions.append(real_action)
							rt1_actions.append(rt1_action)
							print("Got output from head rt1:", i)

					elif name == 'gripper':
						# iterating through the wrist RT1s
						for i in range(self.wrist_rt1_num):
							self.rt1_wrist_observations[i]['image'] = image
							# we are giving each rt1 a slightly different embedding for diversity
							if self.diversify_command_embeddings:
								# we are giving each rt1 a slightly different embedding for diversity
								self.rt1_head_observations[i]['natural_language_embedding'] = self.command_embeddings[i:i+1]
							else:
								self.rt1_head_observations[i]['natural_language_embedding'] = self.average_command_embedding
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
					
				# averaging all the real actions obtained from the RT1s
				real_action = self.average_real_action_outputs_from_multiple_inputs(real_actions)
				record_real_actions(real_action)

				# if RT1 outputs the terminal episode signal, meaning it thinkgs that the task is done, we are plotting the actions and shutting down the node
				is_terminal_episode = real_action['is_terminal_episode']
				if is_terminal_episode == 1.0:
					plot_real_actions()
					plot_rt1_actions()
					
					self.get_logger().info('Terminal Episode. Shutting Down Node...')
					self.destroy_node()
					rclpy.shutdown()
					break

				# publishing rotation
				twist = Twist() 
				twist.linear.x = float(real_action['y'])
				self.twist_pub.publish(twist)

				duration1 = Duration(seconds=10.0)

				head_pan_idx = self.joint_state.name.index('joint_head_pan')
				head_tilt_idx = self.joint_state.name.index('joint_head_tilt')
				wrist_yaw_idx = self.joint_state.name.index('joint_wrist_yaw')
				wrist_pitch_idx = self.joint_state.name.index('joint_wrist_pitch')
				wrist_roll_idx = self.joint_state.name.index('joint_wrist_roll')
				lift_idx = self.joint_state.name.index('joint_lift')
				extension_idx = self.joint_state.name.index('wrist_extension')

				head_pan_value = self.joint_state.position[head_pan_idx]
				head_tilt_value = self.joint_state.position[head_tilt_idx]
				wrist_yaw_value = self.joint_state.position[wrist_yaw_idx]
				wrist_pitch_value = self.joint_state.position[wrist_pitch_idx]
				wrist_roll_value = self.joint_state.position[wrist_roll_idx]
				lift_value = self.joint_state.position[lift_idx]
				extension_value = self.joint_state.position[extension_idx]

				lift_height_remain = LIFT_HEIGHT - lift_value
				angle = np.arctan2(real_action['extension'] + 0.5, lift_height_remain)
				 

				head_pan_pos = -np.pi/2
				head_tilt_pos = -np.pi/2 + angle

				# if we are grasping the object, we are maintaining most of the values but changing the grasping value to closing value
				if need_to_grasp:
					logger.info('Need to grasp triggered')
					positions = [
						lift_value, 
						extension_value+0.05, 
						wrist_yaw_value, 
						wrist_pitch_value,
						wrist_roll_value,
						head_pan_value, 
						head_tilt_value,
						ACTUATOR_GRIPPER_LOWER_LIMIT_CLOSED,
					]
				# if we are not grasping the object, we are using the values from the RT1s
				else:
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
				

				point1 = JointTrajectoryPoint()
				
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

	def image_callback(self, msg):
		"""
		:param msg: the image message coming from the camera

		Everytime we receive a new image message from the robot, this command is triggered and stores the image in the image_state variable
		"""
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
		"""
		:param msg: the image message coming from the camera

		Everytime we receive a new wrist image message from the robot, this command is triggered and stores the image in the image_state variable
		"""
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
		"""
		:param joint_state: the joint state message coming from the robot

		Everytime we receive a new joint state message from the robot, this command is triggered and stores the joint state in the joint_state variable
		"""
		print('joint_state received')
		self.joint_state = joint_state
		self.got_joint_states = True

	
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

		node.main()

		node.destroy_node()
		rclpy.shutdown()
	except KeyboardInterrupt:
		
		node.get_logger().info('Keyboard Interrupt. Shutting Down Node...')
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()