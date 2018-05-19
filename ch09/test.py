#coding:utf-8

import numpy as np
import tensorflow as tf
import gym


def epsilon_greedy(action_distribution, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(np.random.random(action_distribution.shape))
    else:
        return np.argmax(action_distribution)

def discount_rewards(rewards, gamma=0.98):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1): # iterate backwards
        discounted_returns[t] = rewards[t] + discounted_returns[t+1]*gamma
    return discounted_returns

class EpisodeHistory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []

    def add_to_history(self, state, action, reward, state_prime):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)

class Memory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.discounted_returns += episode.discounted_returns

class Agent(object):
    def __init__(
            self, 
            session, 
            state_size,
            action_size,
            hidden_size,
            learning_rate = 1e-3
        ):
        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epsilon = 1

        self.model()
        self.build_training()

    def model(self):
        with tf.variable_scope("model"):
            self.state = tf.placeholder(
                shape=[None, self.state_size],
                dtype=tf.float32
            )
            self.W1 = tf.Variable(tf.random_uniform([self.state_size, self.hidden_size]), name="W1")
            self.b1 = tf.Variable(tf.zeros([self.hidden_size]), name="b1")
            self.layer1 = tf.matmul(self.state, self.W1) + self.b1
            
            self.W2 = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]), name="W2")
            self.b2 = tf.Variable(tf.zeros([self.hidden_size]), name="b2")
            self.layer2 = tf.matmul(self.layer1, self.W2) + self.b2

            self.W3 = tf.Variable(tf.random_uniform([self.hidden_size, self.action_size]), name="W3")
            self.b3 = tf.Variable(tf.zeros([self.action_size]), name="b3")
            self.layer3 = tf.matmul(self.layer2, self.W3) + self.b3

            self.output = tf.nn.softmax(self.layer3)
    
    def build_training(self):
        self.action_input = tf.placeholder(tf.int32, shape=[None])
        self.reward_input = tf.placeholder(tf.float32, shape=[None])

        self.output_index_for_actions = (tf.range(
            0, tf.shape(self.output)[0]) * 
                tf.shape(self.output)[1]) + self.action_input

        # episodeにて選択した行動の行動選択確率
        self.logits_for_actions = tf.gather(
            tf.reshape(self.output, [-1]),
            self.output_index_for_actions
        )

        self.loss = -tf.reduce_mean(
            tf.log(self.logits_for_actions) * self.reward_input
        )

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

        self.check_shape_output = tf.shape(self.output)
        self.check_shape_action_input = tf.shape(self.action_input)
        self.check_shape_reward_input = tf.shape(self.reward_input)
        self.check_shape_output_index = tf.shape(self.output_index_for_actions)
        self.check_shape_logits_index = tf.shape(self.logits_for_actions)

        self.check_range_output_index = tf.range(0, tf.shape(self.output)[0] * tf.shape(self.output)[1])
        self.check_reshape_output = tf.reshape(self.output, [-1])

    def epsilon_decay(self):
        self.epsilon -= 1e-4
        if self.epsilon < 0.1:
            self.epsilon = 0.1

    def predict_action(self, state):
        self.epsilon_decay()

        action_list = self.session.run(
            self.output, 
            feed_dict={self.state: [state]}
        )
        
        # 次元を削減 
        action_distribution = action_list[0]

        # print(np.reshape(action_list, [-1]))

        action = epsilon_greedy(action_distribution, self.epsilon)
        
        return action



def main():
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    total_episodes = 1000
    max_episode_length = 500
    train_frequency = 8
    
    print(state_size)
    print(action_size)

    with tf.Session() as session:
        agent = Agent(
            session = session,
            state_size = state_size,
            action_size = action_size,
            hidden_size = 16
        )

        session.run(tf.global_variables_initializer())

        episode_rewards = []
        batch_losses = []

        global_memory = Memory()

        for i in range(total_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_history = EpisodeHistory()
            for j in range(max_episode_length):
                # env.render()
                action = agent.predict_action(state)
                state_prime, reward, terminal, _ = env.step(action)
                episode_history.add_to_history(
                    state,
                    action,
                    reward,
                    state_prime
                )
                state = state_prime
                episode_reward += reward
                if terminal:
                    episode_history.discounted_returns = discount_rewards(episode_history.rewards)
                    global_memory.add_episode(episode_history)

                    if np.mod(i, train_frequency) == 0:
    
                        feed_dict = {
                            agent.reward_input : np.array(global_memory.discounted_returns),
                            agent.action_input : np.array(global_memory.actions),
                            agent.state : np.array(global_memory.states)
                        }

                        _, batch_loss, = session.run(
                            [agent.train_step, 
                             agent.loss],
                             feed_dict = feed_dict
                        )


                        print("Episode:", i," Reward:", episode_reward," loss:", batch_loss," epsilon:", agent.epsilon)
                        # print("shape output:", session.run(agent.check_shape_output, feed_dict = feed_dict))
                        # print("shape action_input:", session.run(agent.check_shape_action_input, feed_dict = feed_dict))
                        # print("shape reward_input:", session.run(agent.check_shape_reward_input, feed_dict = feed_dict))
                        # print("shape outout_index:", session.run(agent.check_shape_output_index, feed_dict = feed_dict))
                        # print("shape logits_index:", session.run(agent.check_shape_logits_index, feed_dict = feed_dict))

                        # print("range output_index:", session.run(agent.check_range_output_index, feed_dict = feed_dict))

                        print("ouput_index_for_actions:", session.run(agent.output_index_for_actions, feed_dict = feed_dict))
                        print("reshape output:", session.run(agent.check_reshape_output, feed_dict = feed_dict))
                        print("logits_for_actions:", session.run(agent.logits_for_actions, feed_dict = feed_dict))
                        batch_losses.append(batch_loss)
                        global_memory.reset_memory()

                    break

if __name__ == "__main__":
    main()

