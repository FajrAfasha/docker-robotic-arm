import pybullet as p
import numpy as np
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Add a target object
def add_target(position):
    target_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.07),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]),
        basePosition=position
    )
    return target_id


# Add dynamic obstacles
def add_dynamic_obstacle(position):
    obstacle_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[0, 0, 1, 1]),
        basePosition=position
    )
    return obstacle_id

# Place objects and obstacles
def place_objects_on_reach_space(robot_id):
    num_joints = p.getNumJoints(robot_id)
    workspace_points = []

    # Simulate robot movement and collect boundary points
    for joint_1_angle in np.linspace(-np.pi, np.pi, 31):
        for joint_2_angle in np.linspace(-np.pi/3, np.pi/3, 31):
            # for joint_3_angle in np.linspace(-np.pi, np.pi, 10):
              for joint_4_angle in np.linspace(0, np.pi*4/5 , 20):
                    p.resetJointState(robot_id, 0, joint_1_angle)
                    if num_joints > 1:
                        p.resetJointState(robot_id, 1, joint_2_angle) #up and down -pi/2 to + pi/2
                    if num_joints > 2:
                        p.resetJointState(robot_id, 2, np.pi/2) #circle around 2pi
                    if num_joints > 3:
                        p.resetJointState(robot_id, 3, joint_4_angle) #another link up and down 2pi
                    if num_joints > 4:
                        p.resetJointState(robot_id, 4, 0) #circle around 2pi
                    if num_joints > 5:
                        p.resetJointState(robot_id, 5, 0) #circle around 2pi
                    if num_joints > 6:
                        p.resetJointState(robot_id, 6, 0) #circle around 2pi

                    # Get the position of the end effector (last link)
                    end_effector_state = p.getLinkState(robot_id, num_joints - 1)
                    if end_effector_state:
                        end_effector_position = end_effector_state[0]
                        workspace_points.append(end_effector_position)

    # Convert to NumPy array
    workspace_points = np.array(workspace_points)

    # Round positions to avoid duplicates and get unique boundary points
    unique_points = np.unique(np.round(workspace_points, decimals=2), axis=0)

    # Randomly select a target and obstacles from boundary points
    np.random.shuffle(unique_points)
    target_position = unique_points[0]
    obstacle_positions = unique_points[1:2]  # Use next 3 points for obstacles

    target_id = add_target(target_position)

    obstacle_ids = []
    for pos in obstacle_positions:
        obstacle_id = add_dynamic_obstacle(pos)
        obstacle_ids.append(obstacle_id)
    p.resetJointState(robot_id, 0, 0)
    p.resetJointState(robot_id, 1, 0)
    p.resetJointState(robot_id, 2, 0)
    p.resetJointState(robot_id, 3, 0)
    p.resetJointState(robot_id, 4, 0)
    p.resetJointState(robot_id, 5, 0)
    p.resetJointState(robot_id, 6, 0)
    # p.resetJointState(robot_id, 3, 0)
    return target_id, target_position, obstacle_ids, obstacle_positions
