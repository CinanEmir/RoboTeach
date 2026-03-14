import gymnasium as gym
import numpy as np
import pandas as pd
import os

class HumanoidWalkingEnv(gym.Env):
    def __init__(self, render_mode=None, csv_path=None):
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.step_count = 0
        
        self.reference_data = None
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self.reference_data = df.values 
                print(f"STABIL MOD: {len(self.reference_data)} karelik T-Pose takibi aktif.")
            except Exception as e:
                print(f"CSV Hatasi: {e}")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # 1. TEMEL VERILER
        torso_height = obs[2]
        # Humanoid-v5'te obs[28:51] arasi eklem hizlaridir (Velocities)
        joint_velocities = obs[28:51] 

        # 2. ODUL SISTEMI GUNCELLEME
        # Boyu biraz daha esnek ama stabil tutuyoruz
        height_reward = np.exp(-15.0 * (torso_height - 1.20)**2)
        
        # HAYATTA KALMA: Bunu artirdik ki dusmek cok 'pahali' olsun (1-2 sn'den 10-20 sn'ye ciksin)
        alive_reward = 5.0 

        # DURAGANLIK CEZASI: Heykel gibi durmasi icin eklem hizlarini cezalandiriyoruz
        # Robot ne kadar sabit durursa o kadar az ceza yer
        stillness_penalty = -0.01 * np.sum(np.square(joint_velocities))

        # 3. 12 EKLEM TAKLIT (HASSAS MOD)
        weighted_gait_error = 0
        total_weight = 0
        joint_angles = obs[7:28] 
        
        if self.reference_data is not None:
            idx = self.step_count % len(self.reference_data)
            row = self.reference_data[idx]
            
            # Dirsek ve Omuz agirliklarini 20-30 katina cikardik!
            # Artik dirsek bukmesi buyuk bir ceza demek.
            mapping_config = {
                1:  [row[2], 1.0],   # abdomen_y
                2:  [row[3], 1.0],   # abdomen_x
                11: [row[4], 5.0],   # left_hip_y
                5:  [row[5], 5.0],   # right_hip_y
                12: [row[6], 2.0],   # left_knee
                6:  [row[7], 2.0],   # right_knee
                18: [row[10], 30.0], # LEFT SHOULDER (KRITIK)
                15: [row[11], 30.0], # RIGHT SHOULDER (KRITIK)
                20: [row[12], 25.0], # LEFT ELBOW (BUKULMEYI ENGELLE)
                17: [row[13], 25.0], # RIGHT ELBOW (BUKULMEYI ENGELLE)
            }
            
            for r_idx, config in mapping_config.items():
                target_deg, weight = config
                target_rad = np.radians(target_deg)
                error = (joint_angles[r_idx] - target_rad) ** 2
                weighted_gait_error += weight * error
                total_weight += weight
            
            # EXPO Çarpanini 12'den 25'e ciktik (Cok daha keskin bir hedef)
            imitation_reward = np.exp(-25.0 * (weighted_gait_error / total_weight))
        else:
            imitation_reward = 0

        # 4. TOPLAM ODUL
        total_reward = (
            alive_reward + 
            (25.0 * imitation_reward) + # Artik taklit cok daha baskin
            (5.0 * height_reward) + 
            stillness_penalty +         # Titreme cezasi
            -0.001 * np.mean(np.square(action))
        )

        self.step_count += 1
        return obs, total_reward, terminated, truncated, info

    def render(self): return self.env.render()
    def close(self): self.env.close()