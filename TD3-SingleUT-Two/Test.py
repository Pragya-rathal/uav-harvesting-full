import gymnasium as gym
import gym_foo
import numpy as np
import math
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPS = 1e-8
SB3_CUSTOM_OBJECTS = {"lr_schedule": lambda _: 0.0}


def _reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, rewards, terminated, truncated, info = result
        return obs, rewards, terminated or truncated, info
    return result




def _set_train_mode(env, train):
    target = getattr(env, "unwrapped", env)
    target.Train = train
    env.Train = train

def _extract_energy(info, reward):
    if isinstance(info, dict):
        return float(info.get("reward", reward)), float(info.get("received_energy", reward))
    if isinstance(info, (list, tuple)) and info:
        info = info[0]
    if isinstance(info, str):
        harvest_energy, received_energy = info.split(",")[:2]
        return float(harvest_energy), float(received_energy)
    return float(reward), float(reward)
env = gym.make('foo-v0')

model = TD3.load("td3_SingleUT_Two", custom_objects=SB3_CUSTOM_OBJECTS)

obs = _reset_env(env)
_set_train_mode(env, False)
Rewards = []
Harvest = []
Received = []
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = _step_env(env, action)
    harvestEnergy, receivedEnergy = _extract_energy(info, rewards)
    Rewards.append(rewards)
    Harvest.append(harvestEnergy)
    Received.append(receivedEnergy)
    if dones==True:
        break

    env.render()

print(np.sum(Harvest)/max(float(np.sum(Received)), EPS))
np.savetxt("Rewards.csv", Rewards, delimiter=',')
np.savetxt("Total_Reward.txt", [np.sum(Harvest)/max(float(np.sum(Received)), EPS)])
