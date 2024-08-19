# About this project

This is a project I have worked on at the Learning Systems Robotics Lab at the Technical University of Munich during the summer of 2024. The goal of this project was to adopt the RT1 model developed by Google DeepMind and repurpose it to control the Stretch 3 platform developed by Hello Robot. At the end, I was able to give a natural langugage command(e.g. "Pick up the coke can on the table") and the robot was able to execute the task with about 70% success rate. The demo of the project can be found [here](https://youtu.be/U5dWaO7s0pA).

## Main accomplishments

- was able to map the outputs of the RT1 model to the outputs accepted by the stretch 3 robot and also did some mapping from the RT1 frame to the stretch 3 frame
  = Currently the code runs 6 RT1 models all given access to the head camera and the outputs of the models are averaged out to make the process more reliable.
- the grabbing decision come from two sources
  - when one of the RT1 model is confident that it is in a good position to grab the object it will send a signal to the controller to grab the object
  - we are also measuring the cosine similarity between sample images I collected from the wrist camera to the current wrist camera input and if the similarity score exceeds a certain amount then the robot will grab the object. This method has proven to increase the success rate of the robot significantly.

## How to run the code

### If you already have access to the hornywombat laptop used at the LSY lab

run the 3 scripts listed in the `Ken's NOtes.md` file in the `ament_ws` directory of the stretch robot. Then run the following command on the horny wombat laptop:

```bash
cd rt1/rt1_ws/
colcon build
source install/setup.bash
ros2 run rt1_controller rt1_controller
```

### If you want to run the code on your own machine

The clone this repo inside the `src` folder of a ros2 workspace. Then run the following commands:

```bash
colcon build
source install/setup.bash
ros2 run rt1_controller (your alias for the node)
```
