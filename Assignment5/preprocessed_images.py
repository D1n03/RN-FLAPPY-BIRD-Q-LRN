import random
from matplotlib import pylab as plt

import cv2
import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array")

obs, _ = env.reset()
rewards = []
i = 0
while True:
    # Next action:
    # (feed the observation to your agent here)
    obs_image_rgb = env.render()
    image_cropped = obs_image_rgb[:-100, :, :]

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Resize the screen to 84x84
    obs_image_resized = cv2.resize(image_gray, (84, 84))

    i += 1
    print(obs_image_resized)
    if i >= 10:
        plt.imshow(obs_image_resized, cmap="gray")
        plt.show()

    action = 0
    if random.random() < 0.1:
        action = 1

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    rewards.append(reward)

    if terminated:
        obs, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        print(reward)
        print(sum(rewards))
        break

env.close()
