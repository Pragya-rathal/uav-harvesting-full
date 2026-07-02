# Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning
## Introduction
- This repository is the implementation of "Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning". [Paper](https://ieeexplore.ieee.org/document/10051712)
- This study proposed a dual (time and spcace)-domain energy harvesting (EH) approach to maximize EH efficiency of the UAV—RIS system by jointly optimizing the RIS phase shifts vector, the RIS scheduling martix, the length of energy harvesting phase, and the transmit power. 
- For the UAV trajectory design, we considered the density-aware and Fermat point-based algorithms.
- The implementation of DDPG and TD3 using [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/).
- The implementation of SD3 is based on Dr. Pan's research: [Softmax Deep Double Deterministic Policy Gradients](https://github.com/ling-pan/SD3).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me (haoranpeng@cuhk.edu.hk). Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
```
Haoran Peng, and Li-Chun Wang, “Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning”, IEEE Trans. Wireless Commun., vol. 22, no. 10, pp. 6826——6838, Oct., 2023, doi: 10.1109/TWC.2023.3245820 
```
**Bibtex:**
```
  @ARTICLE{10051712,
  author={Peng, Haoran and Wang, Li-Chun},
  journal={IEEE Trans. Wireless Commun.}, 
  title={Energy Harvesting Reconfigurable Intelligent Surface for {UAV} Based on Robust Deep Reinforcement Learning}, 
  year={2023},
  month={Oct.}
  volume={22},
  number={10},
  pages={6826--6838},
 }
```
```
@INPROCEEDINGS{peng1570767WCNC,
  author={Peng, Haoran and Wang, Li-Chun and Li, Geoffrey Ye and Tsai, Ang-Hsun},
  booktitle={Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)}, 
  title={Long-Lasting {UAV}-aided {RIS} Communications based on {SWIPT}},
  address={Austin, TX},
  year={2022},
  month = {Apr.}
}
```
## Requirements (modernized)
- Python: 3.9/3.10
- Pytorch: 1.12+
- gym: 0.26+
- gymnasium (for Stable-Baselines3 v2)
- numpy
- matplotlib
- Stable-Baselines3

## Usage
#### Descriptions of folders
- The folder "XXXX-MultiUT-Time" is the source code for the time-domain EH scheme in the multiple user scenario.
- The folder "XXXX-MultiUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the multiple user scenario.
- The folder "XXXX-SingleUT-Time" is the source code for the time-domain EH scheme in the single user scenario.
- The folder "XXXX-SingleUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the single user scenario.
- The folder "CreateData" is the source code for generating dataset of trajectories files for users and the UAV.

#### Descriptions of files
- For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
- For DRL-based algorithms, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
- You can change the dataset and the scenario in 'gym_foo/envs/foo_env.py'.

#### Training phase (legacy)
1. For the TD3 and DDPG, please execute the TD3.py and DDPG.py to train the model, such as
```
python TD3.py / python DDPG.py
```
***Please change the training mode in the file "gym_foo/envs/foo_env.py" before you executing the training progress.***
For example:
```
class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, LoadData = True, Train = False, multiUT = True, Trajectory_mode = 'Fermat', MaxStep = 41):        
```
If you want to conduct the training phase, the value of "Train" should be "True", otherwise, the value of "Train" should be "Flase" when excuting the testing phase.

2. For the exhaustive search, please execute the ExhaustiveSearch.py to reproduce the simulation results.
3. For the SD3, please execute main.py to train a new model. 

***Legacy scripts were built for Gym 0.15.3. For modern Gym/Stable-Baselines3, use the scripts below.***

## Colab / Kaggle quick start (Gym 0.26+ / Stable-Baselines3)
1. Clone and enter repository:
```
!git clone https://github.com/Haoran-Peng/UAV-RIS_EnergyHarvesting.git
%cd UAV-RIS_EnergyHarvesting
```
2. Install dependencies:
```
!pip install -r requirements.txt
```
3. Train any algorithm (SAC / TD3 / DDPG, default `--timesteps 200000`):
```
!python train_sac.py --timesteps 200000 --model-name sac_colab
!python train_td3.py --timesteps 200000 --model-name td3_colab
!python train_ddpg.py --timesteps 200000 --model-name ddpg_colab
```
4. Run K-sweep evaluation (`K=0.5` and `K=5` by default):
```
# Evaluate one algorithm
!python eval_sweep.py --algo sac --model-path sac_logs/sac_colab.zip

# OR evaluate all three in one line (uses default model paths)
!python eval_sweep.py --algo all
```

Generated files are saved in repo root with algorithm-specific names, e.g.:
- `SINR_vs_Pt_SAC_Kcompare.png`
- `SINR_vs_Pt_TD3_Kcompare.png`
- `SINR_vs_Pt_DDPG_Kcompare.png`
- corresponding `SumRate_*` and combined `SINR_SumRate_*` plots.

### Kaggle clone troubleshooting
If you see `Could not resolve host: github.com`, Kaggle internet is disabled for the notebook.
- Enable Internet in **Notebook settings** and rerun the clone command, or
- Upload this repo as a Kaggle Dataset and copy/extract it locally, e.g.:
```
!cp -r /kaggle/input/<your-dataset-folder>/UAV-RIS_EnergyHarvesting /kaggle/working/
%cd /kaggle/working/UAV-RIS_EnergyHarvesting
```
11. Run the Rician K sweep (weak/medium/strong fading) for TD3/DDPG/SAC and save results:
```
!python rician_k_sweep.py --timesteps 200000 --episodes-per-pt 5 --seed 0 --device cuda
```
12. View sweep plots for a specific K (example K=1):
```
from IPython.display import Image, display

display(Image("results/K_1/sinr_vs_pt.png"))
display(Image("results/K_1/sumrate_vs_pt.png"))
display(Image("results/K_1/training_reward.png"))
```
11. Run the Rician K sweep (weak/medium/strong fading) for TD3/DDPG/SAC and save results:
```
!python rician_k_sweep.py --timesteps 200000 --episodes-per-pt 5 --seed 0 --device cuda
```
12. View sweep plots for a specific K (example K=1):
```
from IPython.display import Image, display

display(Image("results/K_1/sinr_vs_pt.png"))
display(Image("results/K_1/sumrate_vs_pt.png"))
display(Image("results/K_1/training_reward.png"))
```

#### Testing phase
Please execute test.py to evaluate DRL models. Before you produce the testing results, please change the dataset and scenario in 'gym_foo/envs/foo_env.py'.

#### The EH efficiency
The EH efficiency = the harvested energy / the received energy from RF signals


## Publication suite

The repository now includes a `publication_suite/` folder with scripts for:

- 16-way experiment matrix generation
- TD3 ablation tables
- multi-seed robustness sweeps
- multi-user scalability evaluation
- publication plots
- LaTeX table export
- reproducibility reports

See `publication_suite/README.md` for usage.
