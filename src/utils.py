import gym
import pybullet as p
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pybullet_data
from save_test_video import *
import cv2
from ObjectsPlacement import *

# Hyperparameters
EPISODES = 10001
MAX_STEPS = 200
GAMMA = 0.99
LR = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 50000
BATCH_SIZE = 64
distance_thres_epsilon = 1.0
distance_thres_decay = 0.999


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc3(x)
        return x


def plot_list(data, title="Plot of List", xlabel="Index", ylabel="Value"):
    """
    Plots a list using matplotlib.

    Parameters:
    - data (list): The list of numerical values to plot.
    - title (str): Title of the plot (default: "Plot of List").
    - xlabel (str): Label for the x-axis (default: "Index").
    - ylabel (str): Label for the y-axis (default: "Value").
    """
    if not isinstance(data, list):
        raise ValueError("Input must be a list.")

    plt.figure(figsize=(8, 5))  # Set the figure size
    plt.plot(data, marker='o', linestyle='-', color='b')  # Line plot with markers
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  # Add a grid for better readability
    plt.show()
# Environment Wrapper
class RobotArmEnv:
    def __init__(self):
        self.env_init()
        self.joint_angle_arrays = [  # Store the discrete angle arrays for each joint
            np.linspace(-np.pi, np.pi, 31),  # Joint 1 angles
            np.linspace(-np.pi / 3, np.pi / 3, 31),  # Joint 2 angles
            # np.linspace(-np.pi, np.pi, 10),  # Joint 3 angles
            np.linspace(0, np.pi*4/5, 20)
        ]
        self.current_joint_indices = [len(arr) // 2 for arr in self.joint_angle_arrays]  # Start at mid-point
        self.current_joint_indices[-1] = 0
        self.action_space = 3  # Relative steps for [Joint1, Joint2, Joint4]
        self.observation_space = 9  #[Vector to Obstacle, Vector to Target]

    def env_init(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        global robot_id
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

    def reset(self):
        p.resetSimulation()
        self.env_init()
        target_id, self.target_position, self.obstacle_ids, self.obstacle_positions = place_objects_on_reach_space(
            robot_id)
        self.current_joint_indices = [len(arr) // 2 for arr in self.joint_angle_arrays]  # Reset to mid-point
        self.current_joint_indices[-1] = 0
        return self.get_state()

    def step(self, action_steps):
        # Update joint indices based on action steps (-1, 0, +1)
        for i in range(len(self.current_joint_indices)):
            self.current_joint_indices[i] = np.clip(
                self.current_joint_indices[i] + action_steps[i],
                0, len(self.joint_angle_arrays[i]) - 1
            )

        # Get the new joint angles from the updated indices
        new_joint_angles = [
            self.joint_angle_arrays[i][self.current_joint_indices[i]]
            for i in range(len(self.current_joint_indices))
        ]

        # Apply the new joint angles
        for i, joint_index in enumerate([0, 1, 3]):  # Joint indices in the robot
            p.resetJointState(robot_id, joint_index, new_joint_angles[i])
        p.resetJointState(robot_id, 2, np.pi/2)
        p.resetJointState(robot_id, 4, 0)
        p.resetJointState(robot_id, 5, 0)
        p.resetJointState(robot_id, 6, 0)

        # Get the end effector position (last link)
        end_effector_state = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
        end_effector_position = np.array(end_effector_state[0])

        # Calculate distance to the target
        distance_to_target = np.linalg.norm(end_effector_position - self.target_position)
        distance_to_obstacle = np.linalg.norm(end_effector_position - self.obstacle_positions[0])
        if distance_to_obstacle < 0.001:
            reward = -10
            return self.get_state(), reward, True
        reward = -distance_to_target
        global distance_thres_epsilon
        # Check if the task is complete
        done = distance_to_target < 0.001 # * distance_thres_epsilon  # Success if within 1cm of the target

        return self.get_state(), reward, done

    def get_state(self):
        # end_effector_state = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
        # end_effector_position = np.array(end_effector_state[0])
        # return np.concatenate((end_effector_position, self.target_position, self.obstacle_positions[0]))
        end_effector_state = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
        end_effector_position = np.array(end_effector_state[0])
        vector_to_obstacle = self.obstacle_positions[0] - end_effector_position
        vector_to_target = self.target_position - end_effector_position
        return np.concatenate((vector_to_obstacle, vector_to_target, end_effector_position))

# Train DQN agent
def train_dqn(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjust the number of discrete actions to 3 steps per joint (-1, 0, +1)
    num_actions = 3 ** env.action_space
    dqn = DQN(env.observation_space, num_actions).to(device)
    target_dqn = DQN(env.observation_space, num_actions).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    reward = 0
    global distance_thres_decay
    global distance_thres_epsilon
    epsilon = EPSILON_START
    success_rate = 0
    final_reward_jistory = []
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            if random.random() < epsilon:
                action_idx = random.choice(range(num_actions))
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = dqn(state_tensor).detach().cpu().numpy()
                action_idx = np.argmax(q_values)

            # Decode action index into steps (-1, 0, +1 for each joint)
            action_steps = [
                (action_idx // (3 ** i)) % 3 - 1
                for i in range(env.action_space)
            ]

            next_state, reward, done = env.step(action_steps)
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done and reward != -10.:
                print('success', step)
                success_rate += 1

                distance_thres_epsilon *= distance_thres_decay
                distance_thres_epsilon = max(distance_thres_epsilon, 0.1)
                break

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

                # Compute Q-values
                q_values = dqn(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
                next_q_values = target_dqn(next_states_tensor).max(1)[0]
                targets = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)

                # Compute loss and optimize
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        final_reward_jistory.append(reward)
        if episode % 50 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        if episode % 100 == 0 and episode > 0:
            dqn.eval()
            save_test_video(env, dqn, filename=f"test_video_episode_{episode}.avi")
            dqn.train()
            plot_list(final_reward_jistory)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(
            f"Episode {episode}, Total Reward: {total_reward}, success_rate: {success_rate / (episode + 1)}, reward:{reward}, distance_thres_epsilon: {distance_thres_epsilon}")

    plot_list(final_reward_jistory)
    print("Model saved to dqn_robot_arm.pth")


