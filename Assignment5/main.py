import random
from collections import deque

# from packages install opencv-python for cv2
import cv2
import gymnasium
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn.functional as f
from NN import NN

class ReplayBuffer:
    """Replay Buffer for storing and sampling experiences"""
    def __init__(self, capacity):
        # maximum capacity for the buffer
        self.capacity = capacity
        # deque with a fixed size
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, state, action, reward, next_state, done):
        """Add an experience tuple to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def get_length(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class Agent:
    """DQN Agent for Flappy Bird"""
    def __init__(self):
        # CUDA device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # inputs channels: 1 (grayscale), output: 2 (flap or no-flap)
        self.model = NN(1, 2).to(self.device)

        # image preprocessing parameters
        self.resize_width = 84
        self.resize_height = 84

        # training parameters
        self.lr = 0.00001
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_min = 0.0001
        self.decay = 0.995
        # probability of flapping randomly
        self.flap_prob = 0.1

        # replay buffer parameters
        # buffer capacity
        self.capacity = 50000
        # minimum size before training starts
        self.threshold = 5000
        self.replay_buffer = ReplayBuffer(capacity=self.capacity)

        # other training parameters
        self.num_epoches = 100
        self.batch_size = 32
        # mean Squared Error Loss
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # flappy bird environment
        self.env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")
        # flag for loading a pre-trained model
        self.loaded = False

    def change_epsilon(self):
        """Decay epsilon with a lower bound"""
        return max(self.epsilon_min, self.decay * self.epsilon)

    def skip_first_frames(self):
        """Skip frames at the beginning of each episode"""
        for i in range(0, 65):
            if i % 19 == 0:
                # flap sometimes
                _, _, done, _, _ = self.env.step(1)
            else:
                # do nothing
                _, _, done, _, _ = self.env.step(0)
            if done:
                break

    def compute_target_q_values(self, q_values, actions, rewards, next_values, dones):
        """Compute the target Q-values for the Bellman equation"""
        target_values = q_values.clone()
        for i in range(len(target_values)):
            target_values[i][actions[i]] = (rewards[i] + self.gamma * torch.max(next_values[i]) * (1 - dones[i]))

        return target_values

    def preprocess_image(self, obs_image_rgb):
        """Crop, grayscale, and resize the input image"""
        # Remove unnecessary parts of the image
        image_cropped = obs_image_rgb[:-110, :, :]

        # Convert to grayscale
        image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)

        # Resize the image
        obs_image_resized = cv2.resize(image_gray, (self.resize_width, self.resize_height))

        # Normalize pixel values to 0-255 and convert to uint8
        obs_image_normalized = obs_image_resized.astype(np.uint8)

        # Convert to tensor and add a batch dimension
        obs_processed = torch.tensor(obs_image_normalized, dtype=torch.float32).unsqueeze(0)

        return obs_processed

    def train(self, num_epochs=2000):
        """Train the agent for a given number of epochs"""
        epoch = 0
        max_score = 0
        while epoch < num_epochs:
            # save the model every 100 epochs
            if epoch > 1 and (epoch + 1) % 100 == 0:
                self.save_model(f"checkpoints/checkpoint_model2-{9 + (epoch + 1) / 100}.pth")

            epoch += 1
            # Decay epsilon - over time epsilon decreases, reducing random actions in favor of learned actions
            self.epsilon = self.change_epsilon()

            # reset Flappy Bird env for a new episode
            # skip the initial frames of the game to reach a steady state
            self.env.reset()
            self.skip_first_frames()

            # render the current frame of the game environment and display the rendered frame using OpenCV.
            obs_image_rgb = self.env.render()
            cv2.imshow("Flappy Bird", cv2.cvtColor(obs_image_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            # preprocess the RGB image (cropping, resizing, grayscale conversion) and convert it to a PyTorch tensor.
            obs_processed = self.preprocess_image(obs_image_rgb).to(self.device)

            rewards1 = []
            actions1 = []
            loss = 0
            while True:
                # select an action using epsilon-greedy policy
                # if the agent's replay buffer isn't large enough (threshold) or random chance (epsilon) favors exploration:
                # - take a random action (0 for no-flap, 1 for flap) with a probability controlled by flap_prob.
                # otherwise - use the model to predict Q-values for the current state and select the action with the highest Q-value (greedy action).
                if ((self.loaded is False and self.replay_buffer.get_length() < self.threshold)
                        or random.random() < self.epsilon):
                    action = 0
                    if random.random() < self.flap_prob:
                        action = 1  # Random action
                else:
                    with torch.no_grad():
                        q_values = self.model(obs_processed.unsqueeze(0))
                        # Choose the best action
                        action = torch.argmax(q_values).item()

                # Perform the chosen action in the environment
                _, reward, done, _, info = self.env.step(action)
                img = self.env.render()
                cv2.imshow("Flappy Bird", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                # skip additional frames to smooth the agent's behavior, especially when a flap action is taken
                skipped_frames = 1
                if action == 1:
                    skipped_frames = 2
                for i in range(0, skipped_frames):
                    if done is False:
                        _, reward, done, _, info = self.env.step(0)
                        img_rgb = self.env.render()
                        cv2.imshow("Flappy Bird", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                # record the reward and action
                # process the next frame to get the next state representation.
                # store the experience tuple (state, action, reward, next_state, done) in the replay buffer for later training
                rewards1.append(reward)
                actions1.append(action)
                next_image_rgb = self.env.render()
                next_obs_processed = self.preprocess_image(next_image_rgb).to(self.device)
                self.replay_buffer.add_experience(obs_processed, action, reward, next_obs_processed, done)

                # train the model only if the replay buffer has accumulated enough experiences (threshold)
                if self.replay_buffer.get_length() > self.threshold:
                    # sample a batch from the replay buffer
                    batch = self.replay_buffer.sample_batch(self.batch_size)

                    # process the batch
                    # extract components of the sampled batch (states, actions, rewards, next states, done flags)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.stack(states).to(self.device)
                    actions = torch.LongTensor(actions).to(self.device)
                    rewards = torch.Tensor(rewards).to(self.device)
                    next_states = torch.stack(next_states).to(self.device)
                    dones = torch.LongTensor(dones).to(self.device)

                    # forward pass
                    # predicted Q-values for current states
                    # predicted Q-values for next states (detached to avoid gradient computation)
                    q_values = self.model(states)
                    next_q_values = self.model(next_states).detach()

                    # Compute target Q-values using the Bellman equation
                    target_q_values = self.compute_target_q_values(q_values, actions, rewards, next_q_values, dones)

                    # Compute loss
                    loss = f.mse_loss(q_values, target_q_values)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss += loss.item()

                if done:
                    score = info['score']
                    max_score = max(max_score, score)
                    print(f"Epoch: {epoch + 1}, Loss: {loss}, Score: {score}, Best score: {max_score}")
                    print(sum(rewards1), self.replay_buffer.get_length(), self.epsilon)

                    break
                else:
                    obs_processed = next_obs_processed

        self.env.close()

    def save_model(self, filepath):
        """Save the model, optimizer states and the other parameters"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'decay': self.decay,
            'flap_prob': self.flap_prob,
            'capacity': self.capacity,
            'threshold': self.threshold,
            'batch_size': self.batch_size
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.loaded = True
        checkpoint = torch.load(filepath, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_min = checkpoint['epsilon_min']
        self.decay = checkpoint['decay']
        self.flap_prob = checkpoint['flap_prob']
        self.capacity = checkpoint['capacity']
        self.threshold = checkpoint['threshold']
        self.replay_buffer = ReplayBuffer(capacity=self.capacity)
        self.batch_size = checkpoint['batch_size']

    def test(self, n=10):
        max_score = 0
        for j in range(0, n):
            score = 0
            self.env.reset()
            self.skip_first_frames()

            obs_image_rgb = self.env.render()
            cv2.imshow("Flappy Bird", cv2.cvtColor(obs_image_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            obs_processed = self.preprocess_image(obs_image_rgb)

            while True:
                with torch.no_grad():
                    q_values = self.model(obs_processed.unsqueeze(0))
                    action = torch.argmax(q_values).item()

                _, reward, done, _, info = self.env.step(action)
                img = self.env.render()
                cv2.imshow("Flappy Bird", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                skipped_frames = 1
                if action == 1:
                    skipped_frames = 2
                for i in range(0, skipped_frames):
                    if done is False:
                        _, _, done, _, info = self.env.step(0)
                        img_rgb = self.env.render()
                        cv2.imshow("Flappy Bird", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                next_image_rgb = self.env.render()

                next_obs_processed = self.preprocess_image(next_image_rgb)
                if done:
                    score = max(score, info['score'])
                    max_score = max(max_score, score)
                    break
                else:
                    obs_processed = next_obs_processed
            print(f"Score: {score}")
        print(f"Best score: {max_score}")


if __name__ == '__main__':
    agent = Agent()
    agent.load_model("checkpoints/checkpoint_model2-32.0.pth")
    agent.train(2000)
    # agent.test()
    # agent.train(20000)
