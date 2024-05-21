import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np
import subprocess
import time
import os
import multiprocessing.shared_memory as shared_memory
import numpy as np
import sys
import select

# Define a transition tuple to store experience
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#############REMOVE WHEN NOT USED FOR DEBUG#############
def read_camera():
    try:
        # Open the existing shared memory
        existing_shared_mem = shared_memory.SharedMemory(name='/aruco_shared_memory')       

        #shared_array = np.ndarray((5,), dtype=np.float64, buffer=existing_shared_mem.buf)
        posX, posY, posZ, posYaw, Flag = np.ndarray((5,), dtype=np.float64, buffer=existing_shared_mem.buf)
                #close fd and unlink if flag is set to -1, this indicates that the arucopose progam shuting down
        if (Flag == -1):
            existing_shared_mem.close()
            existing_shared_mem.unlink()     

        return posX,posY,posZ,posYaw,Flag
        #print(f"test X = {posX:.0f} y = {posY:.0f} z = {posZ:.0f} yaw = {posYaw:.5f} falg = {Flag:.0f}")

    except FileNotFoundError:
        print("Shared memory does not exist. Please ensure that it is created before running this script.")
    except Exception as e:
        print(f"Error while reading shared memory: {e}")
    existing_shared_mem.close()
    existing_shared_mem.unlink()

class Environment:
    
    def __init__(self):
        self.shared_mem_name = '/aruco_shared_memory'
        self.mavsdk_client = mavsdk.System()
        self.state = None
        self.done = False

    def marker_position(self):
        try:
            # Open the existing shared memory
            existing_shared_mem = shared_memory.SharedMemory(name=self.shared_mem_name)       
            posX, posY, posZ, posYaw, Flag = np.ndarray((5,), dtype=np.float64, buffer=existing_shared_mem.buf)

            if (Flag == -1):
                existing_shared_mem.close()
                existing_shared_mem.unlink()
                self.done = True

            existing_shared_mem.close()
            return np.array([posX, posY, posZ, posYaw]), self.done

        except FileNotFoundError:
            print("Shared memory does not exist. Please ensure that it is created before running this script.")
        except Exception as e:
            print(f"Error while reading shared memory: {e}")

        existing_shared_mem.close()
        existing_shared_mem.unlink()
        return None, True

    def step(self, action):
        velocity = self.action_to_velocity(action)
        self.mavsdk_client.action.set_velocity_body(velocity[0], velocity[1], velocity[2], velocity[3])
        time.sleep(1)  # Wait for the drone to move

        next_state, done = self.marker_position()
        reward = self.compute_reward(next_state)
        return next_state, reward, done, {}

    def reset(self):
        self.done = False
        self.mavsdk_client.action.arm()
        self.mavsdk_client.action.takeoff()
        time.sleep(5)  # Wait for the drone to take off
        state, _ = self.marker_position()
        self.state = state
        return self.state

    def render(self):
        # Implement visualization if needed
        pass

    def close(self):
        self.mavsdk_client.action.land()

    def action_to_velocity(self, action):
        # Define a mapping from action indices to velocity commands
        velocities = [
            (1, 0, 0, 0),  # Move forward
            (-1, 0, 0, 0),  # Move backward
            (0, 1, 0, 0),  # Move left
            (0, -1, 0, 0),  # Move right
            (0, 0, 1, 0),  # Move up
            (0, 0, -1, 0),  # Move down
            (0, 0, 0, 1),  # Rotate clockwise
            (0, 0, 0, -1)  # Rotate counterclockwise
        ]
        return velocities[action]

    def compute_reward(self, state):
        # Example reward function: Negative distance to the target position (e.g., origin)
        target_position = np.array([0, 0, 0, 0])  # Target position and orientation
        distance = np.linalg.norm(state - target_position)
        reward = -distance
        return reward


def start_aruco_pose():
    # Start aruco_pose.py as a subprocess with unbuffered output
    process = subprocess.Popen(
        ["python3", "-u", "aruco_pose.py"],  # -u flag for unbuffered stdout and stderr
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        #text=True,  # Ensure the output is returned as a string
        env=dict(os.environ, PYTHONUNBUFFERED="1")  # Unbuffered environment
    )
    return process


class MemoryDQN:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128) #input layer with 128 neurons
        self.fc2 = nn.Linear(128, 128)        #hidden layer 
        self.fc3 = nn.Linear(128, action_dim)  #outputlayer 

    def forward(self, x):   #forward pass of the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            batch_size=128, 
            gamma=0.99, 
            epsilon_start=1.0, 
            epsilon_end=0.01, 
            epsilon_decay=0.995, 
            target_update=10
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = MemoryDQN(10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0
        self.n_outputs = action_dim

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_outputs)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_model(self):
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def train(env, agent, num_episodes=1000):
    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        for t in range(1000):  # Run for a maximum of 1000 steps
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            if done:
                next_state = None
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            agent.optimize_model()
            agent.update_target_model()
            if done:
                break

def main():
    print("Starting aruco_pose")
    process = start_aruco_pose()
    time.sleep(2)

    # Example usage
    env = Environment()
    n_inputs = 4  # posX, posY, posZ, posYaw
    n_outputs = 8  # 8 possible actions
    agent = Agent(n_inputs, n_outputs)
    train(env, agent)

    process.terminate()
    env.close()

if __name__ == "__main__":
    main()
