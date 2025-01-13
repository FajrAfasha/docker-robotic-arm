import cv2
import pybullet as p
import numpy as np
import torch
import random
def save_test_video(env, model, filename="test_video.avi", fps=5, steps=200):
    # Initialize video writer
    frame_width, frame_height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    reward = 0
    state = env.reset()
    for step in range(steps):
        # Render the environment to capture frames
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0.5],
                                                          distance=2.5, yaw=0, pitch=-30, roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        _, _, px, _, _ = p.getCameraImage(width=frame_width, height=frame_height,
                                          viewMatrix=view_matrix,
                                          projectionMatrix=proj_matrix,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
        frame = np.array(px, dtype=np.uint8).reshape((frame_height, frame_width, 4))
        frame = frame[:, :, :3]  # Remove alpha channel
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(frame)

        # Select action using the model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        q_values = model(state_tensor).detach().cpu().numpy()
        action_idx = np.argmax(q_values)

        action_steps = [
            (action_idx // (3 ** i)) % 3 - 1
            for i in range(env.action_space)  # env.action_space should be 3 for [Joint1, Joint2, Joint3]
        ]

        # Take a step in the environment using the relative action steps
        state, reward, done = env.step(action_steps)

        if done :
            print("success in video")
            break

    video_writer.release()
    print(f"Test video saved as {filename} with reward {reward}")
