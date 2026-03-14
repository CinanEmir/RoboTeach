Humanoid AI: Pose Imitation & Balance Control
This project focuses on training a Humanoid agent to perform complex pose imitation (T-Pose) while maintaining physical stability in a high-fidelity simulation environment. Utilizing Reinforcement Learning (PPO) and MuJoCo physics engine, the agent learns to bridge the gap between human motion data and robotic torque constraints.

🚀 Key Features
Deep Reinforcement Learning: Powered by Stable Baselines3, implementing Proximal Policy Optimization (PPO) with custom curriculum learning strategies.

Pose Imitation (MoCap-to-Sim): Translates MediaPipe-extracted human pose coordinates into robotic joint targets through a custom-mapped reward function.

Dynamic Balance System: Advanced reward engineering focusing on Center of Mass (CoM) stabilization, joint smoothness, and ankle torque modulation to prevent falling.

Humanoid Command Center: A professional PyQt6 GUI designed for:

One-click training initialization and fine-tuning.

Real-time GPU/System monitoring.

Integrated Model Explorer for visualizing checkpoints.

Live log analysis and process management.

🛠 Tech Stack
Engine: MuJoCo 3.0+

Environment: Gymnasium (Humanoid-v5)

AI Framework: Stable Baselines3 / PyTorch

GUI: PyQt6 / QtAwesome

Data Processing: MediaPipe, NumPy, Pandas
