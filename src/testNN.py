import gym
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

checkpoint_path = './checkpoints/cp.ckpt' # file to store the model

if __name__ == '__main__':
    TEST_EPOCHS = 100

    if os.path.exists('./checkpoints'):
        model = tf.keras.models.load_model(checkpoint_path)
        model.summary()
    else:
        print("No model found\n")
        exit(0)

    env = gym.make('MountainCar-v0', render_mode='human')

    maxmax = np.array([])
    victories = 0

    for epoch in range(1, TEST_EPOCHS + 1):
        obs, _ = env.reset()
        
        # add the first observation of the environment
        maxmax = np.append(maxmax, obs[0])

        c_reward = 0

        # in order to start the loop
        terminated = False
        truncated = False

        while not terminated and not truncated:
            act = np.argmax(model.predict_on_batch(tf.constant([obs]))[0])

            obs_next, reward, terminated, truncated, _ = env.step(act) # execute action and collect corresponding info

            c_reward += reward   # cumulate reward (for evaluation only)

            obs = obs_next # update current state

            # update the actual value of maxmax if the observation has an higher value
            if obs[0] > maxmax[epoch - 1]:
                maxmax[epoch - 1] = obs[0]

            # terminated means that the car has reached the goal
            # truncation means that the episode has reached 200
            if (terminated or truncated):
                if terminated:
                    victories += 1
                
                # print debug information
                print("Epoch:", epoch, "/", TEST_EPOCHS, "--- Score:", c_reward, "--- Victories:", victories, "--- Current value:", maxmax[epoch - 1])

    env.close()

    plt.plot(np.arange(1, TEST_EPOCHS + 1), maxmax)
    plt.show()