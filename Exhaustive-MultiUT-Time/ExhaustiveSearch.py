import numpy as np
import globe
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_foo

EPS = 1e-8

def plot(frame_idx, rewards):
    plt.figure()
    plt.title('Step %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig('Exhaustive_Result/rewards.png', format='png')
    plt.close()

##########################################################################
maxStep = 41

globe._init()
env = gym.make('foo-v0')
if hasattr(env, "seed"):
	env.seed(100)

total_record = []
def main():
	rewards = []
	received_energy_per_step = []
	harvested_energy_per_step = []

	for steps in range(maxStep):
		print("The current steps: "+str(steps))
		max_reward = 0
		harvest_energy = 0
		energy_per_step = 0
		action = []
		for i in range(int(1e6)):
			action = np.random.rand(20)
			next_state, reward, done, received_energy = env.step(action, steps)
			reward_ratio = reward / max(float(received_energy), EPS)
			if reward_ratio > max_reward:
				max_reward = reward_ratio
				energy_per_step = received_energy
				harvest_energy = reward

		rewards.append(max_reward)
		received_energy_per_step.append(energy_per_step)
		harvested_energy_per_step.append(harvest_energy)
		if steps == maxStep - 1:
			np.savetxt("Exhaustive_Result/rewards.csv", rewards, delimiter=',')
			np.savetxt("Exhaustive_Result/received_energy_per_step.csv", received_energy_per_step, delimiter=',')
			np.savetxt("Exhaustive_Result/harvested_energy_per_step.csv", harvested_energy_per_step, delimiter=',')

	total_record.append(np.sum(rewards))

	print("rewards："+str(total_record[0]))
	print("harvested rewards："+str(np.sum(harvested_energy_per_step)/max(float(np.sum(received_energy_per_step)), EPS)))

	np.savetxt("Exhaustive_Result/total_rewards.csv", total_record, delimiter=',')

if __name__ == '__main__':
	main()