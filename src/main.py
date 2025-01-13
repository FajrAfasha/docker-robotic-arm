import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from utils import *
if __name__ =="__main__":
    # Define discrete actions (e.g., discretized joint angles)
    env = RobotArmEnv()
    train_dqn(env)
