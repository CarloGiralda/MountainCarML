import gym
import random
import numpy as np
import matplotlib.pyplot as plt

class Q_Learning:
    def __init__(self, env, epsilon_max, epsilon_min, decay_rate, pos_acc, vel_acc, gamma):
        self.env = env
        self.eps = epsilon_max
        self.eps_min = epsilon_min
        self.dr = decay_rate
        self.gamma = gamma
        # - there are 18 (from -1.2 to 0.6) units of space
        #   each unit can be divided into 100 subunits, so we round the position to 3 decimals
        #   we obtain 18 * 100 = 1 800 values at most for position
        # - there are 14 (from -0.07 to 0.07) units of velocity
        #   each unit can be divided into 10 subunits, so we round the velocity to 3 decimals
        #   we obtain 14 * 10 = 140 values at most for velocity
        # - there are 3 actions
        self.pos_acc = max(pos_acc, 1) # less than 1 is not accepted
        self.vel_acc = max(vel_acc, 2) # less than 2 is not accepted
        self.pos_space = self.env.observation_space.high[0] - self.env.observation_space.low[0]
        self.vel_space = self.env.observation_space.high[1] - self.env.observation_space.low[1]
        self.Q_table = np.zeros((int(self.pos_space * pow(10, self.pos_acc)), int(self.vel_space * pow(10, self.vel_acc)), self.env.action_space.n))

    def act(self, state):
        # pick random action (default)
        act = self.env.action_space.sample()
        if random.randint(0, 100) >= self.eps:
            pos = self.mapToPos(state)
            vel = self.mapToVel(state)
            list_of_act = self.Q_table[pos, vel]
            act = np.argmax(list_of_act)
        return act
    
    def mapToPos(self, state):
        pos = int(round((state[0] - self.env.observation_space.low[0]) * pow(10, self.pos_acc), 0))
        return pos

    def mapToVel(self, state):
        vel = int(round((state[1] - self.env.observation_space.low[1]) * pow(10, self.vel_acc), 0))
        return vel

    def store(self, state, action, reward, next_state):
        # map next_state to a position in the array
        pos_next = self.mapToPos(next_state)
        vel_next = self.mapToVel(next_state)
        # pick the array with the actions for that state
        list_of_act = self.Q_table[pos_next, vel_next]
        # pick the highest value among the q-values for the state
        current = np.amax(list_of_act)

        pos = self.mapToPos(state)
        vel = self.mapToVel(state)
        self.Q_table[pos, vel, action] = reward + self.gamma * current

    def adjust(self):
        self.eps = max(self.eps_min, self.eps * self.dr)
        return self.eps

if __name__ == '__main__':
    EPS_MAX = 100 # Initial exploration probability
    EPS_MIN = 1 # Final exploration probability
    DECAY_RATE = 0.95 # decay rate
    GAMMA = 0.99 # discount factor
    POS_ACCURACY = 3 # decimals taken into consideration for position (NO LESS THAN 1)
    VEL_ACCURACY = 3 # decimals taken into consideration for velocity (NO LESS THAN 2)
    NUM_OF_EPOCHS = 100000 # number of epochs the model should be trained on

    env = gym.make('MountainCar-v0')

    maxmax = np.array([])

    q_learn = Q_Learning(env, EPS_MAX, EPS_MIN, DECAY_RATE, POS_ACCURACY, VEL_ACCURACY, GAMMA)

    victories = 0
    eps = EPS_MAX

    for epoch in range(1, NUM_OF_EPOCHS + 1):
        obs, _ = env.reset()
        
        # add the first observation of the environment
        maxmax = np.append(maxmax, obs[0])

        c_reward = 0

        # in order to start the loop
        terminated = False
        truncated = False

        while not terminated and not truncated:
            state = np.array([round(obs[0], POS_ACCURACY), round(obs[1], VEL_ACCURACY)])
            act = q_learn.act(state)

            obs_next, reward, terminated, truncated, _ = env.step(act) # execute action and collect corresponding info
            state_next = np.array([round(obs_next[0], POS_ACCURACY), round(obs_next[1], VEL_ACCURACY)])

            q_learn.store(state, act, reward, state_next)

            c_reward += reward # cumulate reward (for evaluation only)

            obs = obs_next # update current state

            # update the actual value of maxmax if the observation has an higher value
            if obs[0] > maxmax[epoch - 1]:
                maxmax[epoch - 1] = obs[0]

            # terminated means that the car has reached the goal
            # truncation means that the episode has reached 200
            if (terminated or truncated):
                if epoch % 5 == 0:
                    # reduce the exploration rate
                    eps = q_learn.adjust()

                if terminated:
                    victories += 1
                
                # print debug information
                print("Epoch:", epoch, "/", NUM_OF_EPOCHS, "--- Score:", c_reward, "--- Epsilon:", eps, "--- Victories:", victories, "--- Current value:", maxmax[epoch - 1])
        
    env.close()
    plt.plot(np.arange(1, NUM_OF_EPOCHS + 1), maxmax)
    plt.show()