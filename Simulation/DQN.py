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

def read_shared_memory():
    try:
        # Open the existing shared memory
        existing_shared_mem = shared_memory.SharedMemory(name='/aruco_shared_memory')
        
        while True:

            # Access the shared memory as a numpy array
            shared_array = np.ndarray((5,), dtype=np.float64, buffer=existing_shared_mem.buf)            
            if (shared_array[-1] > 0) or (shared_array[-1] < 0):
                # Print the values stored in shared memory
                print(f"Shared Memory Values - X: {shared_array[0]:.1f}, Y: {shared_array[1]:.1f}, Z: {shared_array[2]:.1f}, Yaw: {shared_array[3]:.1f}, Flag: {shared_array[-1]}")
                # Check for user input or termination signal
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0] or (shared_array[-1] == -1).any():
                    if shared_array[-1] == -1:
                        print("Sender stops sending: Quitting")
                    else:
                        print("User input 'q' or termination signal received: Quitting")
                    break
            elif(shared_array[-1]==0):
                print(f"Not target, Flag: {shared_array[-1]:.0f}")
            else:
                print("ERROR")
 
    except FileNotFoundError:
        print("Shared memory does not exist. Please ensure that it is created before running this script.")
    except Exception as e:
        print(f"Error while reading shared memory: {e}")
    existing_shared_mem.close()
    existing_shared_mem.unlink()
    

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




class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(
            self, 
            n_inputs, 
            n_outputs, 
            batch_size=128, 
            gamma=0.99, 
            epsilon_start=1.0, 
            epsilon_end=0.01, 
            epsilon_decay=0.995, 
            target_update=10
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_inputs, n_outputs).to(self.device)
        self.target_net = DQN(n_inputs, n_outputs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0
        self.n_outputs = n_outputs

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
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()
            agent.update_target_model()
            if done:
                break

def main():
    print("Starting aruco_pose")
    process = start_aruco_pose()
    time.sleep(2)
    
    try:
        read_shared_memory()
    except KeyboardInterrupt:
        print("Stopping aruco_pose")
        time.sleep(1)
        shared_memory
        process.terminate()
        process.wait()
        print("aruco_pose terminated")
    except Exception as e:
        print(f'An error occurred: {e}')
        process.terminate()
        process.wait()
        print("aruco_pose terminated due to error")


if __name__ == "__main__":
    # Example usage:
    # Create your environment and agent
    # env = ...
    # n_inputs = env.observation_space.shape[0]
    # n_outputs = env.action_space.n
    # agent = Agent(n_inputs, n_outputs)
    # Train the agent
    # train(env, agent)

    # In this example, let's say we have an environment with 4-dimensional observations and 2 actions
    n_inputs = 4
    n_outputs = 5
    agent = Agent(n_inputs, n_outputs)
    # You need to provide your environment to the train function
    # train(env, agent)