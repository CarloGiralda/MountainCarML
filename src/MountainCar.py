import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

checkpoint_path = './checkpoints/cp.ckpt' # file to store the model

class Deep_Q_Learning:
    def __init__(self, env, gamma, exp_size, batch_size, epsilon_max, epsilon_min, learning_rate, decay_rate, min_exp):
        self.env = env
        self.gamma = gamma
        self.exp_size = exp_size
        # Past experience arranged as a queue
        self.experience = deque(maxlen=self.exp_size)
        self.batch_size = batch_size
        # size of the observation space (2)
        self.state_size = self.env.observation_space.shape[0]
        # size of the action space (3)
        self.action_size = self.env.action_space.n
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.dr = decay_rate
        self.min_exp = min_exp
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, input_shape=(2,), activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        # optimizer is SGD, stochastic gradient descent, while loss is MSE, mean squared error
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=tf.keras.losses.MeanSquaredError())
        return model
    
    def act(self, state):
        # pick random action (default)
        act = self.env.action_space.sample()
        if random.randint(0, 100) >= self.epsilon:
            act = np.argmax(self.model.predict_on_batch(tf.constant([state]))[0])
        return act

    def store(self, state, next_state, action, reward, terminated, truncated):
        self.experience.append([state, next_state, action, reward, terminated, truncated])

    def learn(self):
        if len(self.experience) >= self.min_exp:
            batch = random.sample(self.experience, self.batch_size)

            state = np.zeros((self.batch_size, self.state_size))
            next_state = np.zeros((self.batch_size, self.state_size))
            action, reward, terminated, truncated = [], [], [], []

            for i in range(self.batch_size):
                state[i] = batch[i][0]
                next_state[i] = batch[i][1]
                action.append(batch[i][2])
                reward.append(batch[i][3])
                terminated.append(batch[i][4])
                truncated.append(batch[i][5])

            target = np.zeros((self.batch_size, self.action_size))
            target_next = np.zeros((self.batch_size, self.action_size))
            for st in range(self.batch_size):
                target[st] = self.model.predict_on_batch(tf.constant([state[st]]))[0]
                target_next[st] = self.model.predict_on_batch(tf.constant([next_state[st]]))[0]

            for i in range(self.batch_size):
                if terminated[i] or truncated[i]:
                    target[i][action[i]] = reward[i]
                else:
                    target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])

            target = np.array(target)

            self.model.fit(tf.constant(state), tf.constant(target), verbose=0)

    def adjust(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.dr) # reduce epsilon by decay rate
        return self.epsilon

    def save(self):
        tf.keras.models.save_model(self.model, checkpoint_path)

if __name__ == '__main__':
    EXP_MAX_SIZE = 20000 # truncation of experience
    MIN_EXP = 2000 # minimum experience to train
    BATCH_SIZE = 64 # size of the training set (used to avoid local minima)
    EPS_MAX = 100 # Initial exploration probability
    EPS_MIN = 1 # Final exploration probability
    GAMMA = .99 # discount factor
    LR = 0.001 # Final learning rate
    DECAY_RATE = 0.9 # decay rate
    NUM_OF_EPOCHS = 5000 # number of epochs the model should be trained on
    eps = EPS_MAX

    maxmax = np.array([])

    env = gym.make('MountainCar-v0')
    
    deep_q = Deep_Q_Learning(env, GAMMA, EXP_MAX_SIZE, BATCH_SIZE, EPS_MAX, EPS_MIN, LR, DECAY_RATE, MIN_EXP)

    victories = 0
    exp = 0

    for epoch in range(1, NUM_OF_EPOCHS + 1):
        obs, _ = env.reset()
        
        # add the first observation of the environment
        maxmax = np.append(maxmax, obs[0])

        c_reward = 0

        # in order to start the loop
        terminated = False
        truncated = False

        while not terminated and not truncated:
            act = deep_q.act(obs)

            obs_next, reward, terminated, truncated, _ = env.step(act) # execute action and collect corresponding info

            deep_q.store(obs, obs_next, act, reward, terminated, truncated)

            c_reward += reward   # cumulate reward (for evaluation only)

            obs = obs_next # update current state

            if c_reward % 10 == 0:
                deep_q.learn()

            # update the actual value of maxmax if the observation has an higher value
            if obs[0] > maxmax[epoch - 1]:
                maxmax[epoch - 1] = obs[0]

            # terminated means that the car has reached the goal
            # truncation means that the episode has reached 200
            if (terminated or truncated):
                eps = deep_q.adjust()
                if epoch % 50 == 0:
                    deep_q.save()

                if terminated:
                    victories += 1

                # print debug information
                print("Epoch:", epoch, "/", NUM_OF_EPOCHS, "--- Score:", c_reward, "--- Epsilon:", eps, "--- Win:", victories, "--- Current value:", maxmax[epoch - 1])
                
    env.close()

    plt.plot(np.arange(1, NUM_OF_EPOCHS + 1), maxmax)
    plt.show()