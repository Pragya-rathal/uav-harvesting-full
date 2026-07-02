import gymnasium as gym
from gym import error, spaces, utils
from gym.utils import seeding
import globe
import numpy as np
import random as rd
import time
import math as mt
import sys
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, readDistance = True, filename="Train.csv", MaxStep = 81):
        globe._init()
        #the location of UAV-RIS
        globe.set_value('L_U', [80, 80, 20]) #[x, y, z]
        #the location of AP/BS
        globe.set_value('L_AP', [0, 0, 10])

        #CSI parameters
        globe.set_value('BW', mt.pow(10, 7)) #The bandwidth is 10 MHz
        # Noise power spectrum density is -174dBm/Hz;
        globe.set_value('N_0', mt.pow(10, ((-174 / 3) / 10)))
        globe.set_value('Xi', mt.pow(10, (3/10))) #the path loss at the reference distance D0 = 1m, 3dB;
        # urban env. from [Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks] 
        #and [Joint Trajectory-Task-Cache Optimization with Phase-Shift Design of RIS-Assisted UAV for MEC]
        globe.set_value('a', 9.61)
        globe.set_value('b', 0.16)
        globe.set_value('eta_los', 1) 
        globe.set_value('eta_nlos', 20)

        # σ2 = -102dBm
        globe.set_value('AWGN', mt.pow(10, (-102/10)))
        # number of RIS antenna
        globe.set_value('N_ris', 100)
        #energy harvesting efficiency eta=0.7
        globe.set_value('eta', 0.7) 
        #path-loss exponent is α=3
        globe.set_value('alpha', 3)
        # additional attenuation factor φ is 20 dB
        globe.set_value('varphi', mt.pow(10, (20/10)))
        # max transmit Power from BS is 500W
        globe.set_value('P_max', 5 * mt.pow(10, 5))#mt.pow(10, 43/10)
        #number of user 
        globe.set_value('N_u', 3)
        # carrier frequency is 750 MHz
        globe.set_value('fc', 750 * mt.pow(10, 6))
        # speed of light
        globe.set_value('c', 3 * mt.pow(10, 8))
        #minimal requirement of sinr 12db
        globe.set_value('gamma_min', mt.pow(10, (12/10)))
        # the transmission power from the AP for a single user
        globe.set_value('power_i', 0.5 * mt.pow(10, 3))
        # the length of a time slot
        globe.set_value('t', int(MaxStep))
        #current time slot
        globe.set_value('step', 0)

        globe.set_value('kappa', mt.pow(10, (-30/10)))
        globe.set_value('hat_alpha', 2.5)
        globe.set_value('successCon', 0)

        if readDistance == True:
            p = filename
            with open(p, encoding = 'utf-8') as f:
                data = np.loadtxt(f, delimiter = ",")
                data.astype(int)
                globe.set_value('DistanceRU', data)
        # else :
            # globe.set_value('DistanceRU', self.D_RU(globe.get_value('t')))
        
        self.action_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(20, 60, shape=(3,), dtype=np.float32)
        self._max_episode_steps = 81

    def step(self, action):
        t = globe.get_value('t')
        tau = action[0]

        lamda = action[1]

        step = globe.get_value('step')
        reward, radio_state, HarvestEnergy = self.env_state(step, tau, lamda)
        done = False
        if step == t - 1:
            done = True
   
        globe.set_value('step', int(step+1))

        return radio_state, reward, done, HarvestEnergy

    def reset(self):
        globe.set_value('step', 0)
        globe.set_value('successCon', 0)
        d_ru = globe.get_value('DistanceRU')
        next_dru = d_ru[0]
        radio_state = np.array([next_dru])#np.array([globe.get_value('power_i'), next_dru])
        return radio_state    

    def _flight_energy(self, delta):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        if delta.size < 3:
            delta = np.pad(delta, (0, 3 - delta.size), constant_values=0.0)
        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        alpha_xy = 1.0
        alpha_z = 1.5
        return alpha_xy * (dx * dx + dy * dy) + alpha_z * (dz * dz)

    def _csi_perturb(self, link):
        link = np.asarray(link)
        if self.csi_mode != "imperfect":
            return link
        scale = max(float(self.sigma_csi), EPS)
        if np.iscomplexobj(link):
            noise = (np.random.normal(size=link.shape) + 1j * np.random.normal(size=link.shape)) * (scale / np.sqrt(2.0))
            return (link + noise).astype(np.complex128)
        noise = np.random.normal(scale=scale, size=link.shape)
        return (link + noise).astype(link.dtype if hasattr(link, "dtype") else np.float32)

    def _build_observation(self, step_idx, radio_state, reward, received_energy):
        radio_state = np.asarray(radio_state, dtype=np.float32).reshape(-1)
        if radio_state.size < 2:
            radio_state = np.pad(radio_state, (0, 2 - radio_state.size), constant_values=0.0)
        reward = float(reward)
        received_energy = float(received_energy)
        step_den = max(float(globe.get_value("t") - 1), 1.0)
        lu_val = globe.get_value("L_U")
        if lu_val is None:
            lu_val = self.uav_position
        L_U = np.asarray(lu_val, dtype=np.float32).reshape(-1)
        if L_U.size < 3:
            L_U = np.pad(L_U, (0, 3 - L_U.size), constant_values=0.0)

        if self.observation_mode == "enhanced":
            base = [
                np.array([float(step_idx) / step_den], dtype=np.float32),
                L_U[:3].astype(np.float32),
                radio_state[:2].astype(np.float32),
                np.array(
                    [
                        reward,
                        received_energy,
                        1.0 if self.trajectory_mode == "learned" else 0.0,
                        1.0 if self.csi_mode == "imperfect" else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
            extras = []
            for key in self._user_keys:
                user = globe.get_value(key)
                if user is None:
                    user = np.zeros(3, dtype=np.float32)
                user = np.asarray(user, dtype=np.float32).reshape(-1)
                if user.size < 3:
                    user = np.pad(user, (0, 3 - user.size), constant_values=0.0)
                dist = float(np.linalg.norm(L_U[:3] - user[:3]))
                extras.append(np.array([user[0], user[1], user[2], dist], dtype=np.float32))
            obs = np.concatenate(base + extras).astype(np.float32)
            if obs.size < self._enhanced_obs_dim:
                obs = np.pad(obs, (0, self._enhanced_obs_dim - obs.size), constant_values=0.0)
            return obs[: self._enhanced_obs_dim].astype(np.float32)

        return np.array([radio_state[0], radio_state[1], reward, received_energy], dtype=np.float32)

    def render(self, mode='human', close=False):
        pass

    def pl_BR(self):
        L_U = globe.get_value('L_U')
        L_AP = globe.get_value('L_AP')
        a = globe.get_value('a')
        b = globe.get_value('b')
        varphi = globe.get_value('varphi')
        alpha = globe.get_value('alpha')

        theta = (180 / mt.pi) * mt.asin( ( (L_U[2] - L_AP[2]) / mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2))) )
        p_los = 1 + a * mt.exp(a * b - b * theta )
        p_los = 1 / p_los

        p_nlos = 1 - p_los
        # channel power gain (BS-RIS) with the los and nlos
        g_BR = (p_los + p_nlos * varphi) * mt.pow(mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2)), (0-alpha))
        
        return g_BR

    def EH(self, tau, lamda):
        power_i = globe.get_value('power_i')
        eta = globe.get_value('eta')

        g_BR = self.pl_BR()
        power_total = power_i * globe.get_value('N_u')
        E_t = tau * eta * power_total * g_BR + (1 - tau) * (1 - lamda) * eta * power_total * g_BR

        return E_t
        
    def capacity (self, distance, tau, lamda):
        kappa = globe.get_value('kappa')
        hat_alpha = globe.get_value('hat_alpha')
        power_i = globe.get_value('power_i')
        AWGN = globe.get_value('AWGN')
        BW = globe.get_value('BW')

        SINR = []
        for i in range(0, globe.get_value('N_u')):

            g_BR = self.pl_BR()
            signal =  power_i * g_BR * lamda * kappa * mt.pow((distance[i]/1), -hat_alpha) * (1 - tau)
            print(g_BR)

            if signal > 0:
                SINR.append(10 * mt.log((signal/AWGN), 10)) #(interference**2)+
            else:
                SINR.append(0)

        return SINR

    def env_state(self, step, tau, lamda):
        d_ru = globe.get_value('DistanceRU')
        if step < globe.get_value('t')-1:
            next_dru = d_ru[step+1]
        else:
            next_dru = d_ru[step]

        reward = self.EH(tau, lamda)
        HarvestEnergy = reward
        SINR = self.capacity (d_ru[step], tau, lamda)
        decline = reward * 1/3
        for i in range(len(SINR)):
            if SINR[i] < 12:
                reward = reward - decline
                HarvestEnergy = 0

        reward = reward*pow(2, (reward/decline))
        radio_state = next_dru#np.array([next_dru])

        return reward, radio_state, HarvestEnergy

    def reloadData(filename):
        with open(filename, encoding = 'utf-8') as f:
            data = np.loadtxt(f, delimiter = ",")
            data.astype(int)
            globe.set_value('DistanceRU', data)
