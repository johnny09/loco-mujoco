# PND Adam Lite Robot

## Overview

The PND Adam Lite is a humanoid robot with 25 degrees of freedom (excluding the floating base):
- **Legs**: 6 DOF per leg (hip pitch/roll/yaw, knee pitch, ankle pitch/roll)
- **Waist**: 3 DOF (roll, pitch, yaw)
- **Arms**: 5 DOF per arm (shoulder pitch/roll/yaw, elbow, wrist yaw)

## Motion Retargeting Support

This robot model includes 15 **mimic sites** for motion retargeting from motion capture data:

### Sites Configuration
- `pelvis_mimic` - Pelvis/root tracking
- `left_hip_mimic`, `left_knee_mimic`, `left_foot_mimic` - Left leg
- `right_hip_mimic`, `right_knee_mimic`, `right_foot_mimic` - Right leg
- `upper_body_mimic`, `head_mimic` - Torso and head
- `left_shoulder_mimic`, `left_elbow_mimic`, `left_hand_mimic` - Left arm
- `right_shoulder_mimic`, `right_elbow_mimic`, `right_hand_mimic` - Right arm

## Usage

### Using Default Dataset (Recommended - 立即可用)

默认数据集包含手工制作的动作，所有机器人都可以直接使用：

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

env = ImitationFactory.make(
    "PndAdamLite",
    default_dataset_conf=DefaultDatasetConf(["squat", "walk", "run"]),
    n_substeps=20
)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

### Using LAFAN1 Dataset (需要预处理)

⚠️ **注意**: LAFAN1 数据需要为每个机器人预先转换。新机器人需要：

1. **方案A**: 从其他机器人迁移（robot-to-robot retargeting）

```python
from loco_mujoco.smpl.retargeting import retarget_traj_from_robot_to_robot
from loco_mujoco.datasets.humanoids.LAFAN1 import load_lafan1_trajectory
from loco_mujoco.task_factories import ImitationFactory

# 从UnitreeG1加载LAFAN1数据（已有预处理数据）
traj_source = load_lafan1_trajectory("UnitreeG1", ["walk1_subject1"])

# 将轨迹重定向到PndAdamLite
traj_target = retarget_traj_from_robot_to_robot(
    env_name_source="UnitreeG1",
    traj_source=traj_source,
    env_name_target="PndAdamLite"
)

# 使用重定向后的轨迹
env = ImitationFactory.make("PndAdamLite", n_substeps=20)
env.load_trajectory(traj_target)
env.play_trajectory(n_episodes=3, render=True)
```

2. **方案B**: 联系 LocoMuJoCo 团队，请求为 PndAdamLite 生成预处理数据

### Using AMASS Dataset (需要SMPL安装)

```python
from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf

env = ImitationFactory.make(
    "PndAdamLite",
    amass_dataset_conf=AMASSDatasetConf([
        "DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"
    ]),
    n_substeps=20
)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

**要求**: 需要安装 SMPL 依赖，参见 `loco_mujoco/smpl/README.MD`

## 数据集对比

| 数据集 | 格式 | PndAdamLite支持 | 说明 |
|--------|------|----------------|------|
| **Default** | 内置 | ✅ 立即可用 | 手工制作，质量高 |
| **LAFAN1** | BVH→npz | ⚠️ 需要转换 | 需要从其他机器人迁移或预处理 |
| **AMASS** | SMPL | ✅ 可用 | 需要安装SMPL，数据量大 |

## 推荐使用流程

### 新手/快速测试
```python
# 使用 Default 数据集 - 无需额外配置
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

env = ImitationFactory.make(
    "PndAdamLite",
    default_dataset_conf=DefaultDatasetConf(["walk", "run"]),
    n_substeps=20
)
```

### 需要更多数据/训练
```python
# 使用 AMASS 数据集 - 需要安装SMPL
from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf

env = ImitationFactory.make(
    "PndAdamLite",
    amass_dataset_conf=AMASSDatasetConf(["KIT/12/WalkInClockwiseCircle11_poses"]),
    reward_type="MimicReward",
    goal_type="GoalTrajMimic",
    n_substeps=20
)
```

## Example Script

运行示例脚本（使用Default数据集）:
```bash
python examples/replay_datasets/pnd_adam_lite_example.py
```

## Configuration

重定向配置位于 `loco_mujoco/smpl/robot_confs/PndAdamLite.yaml`:

```yaml
robot_pose_modifier:
  - shoulderRoll_Left: "np.pi/2"
  - elbow_Left: "-np.pi/4"
  - shoulderRoll_Right: "-np.pi/2"
  - elbow_Right: "-np.pi/4"

optimization_params:
  z_offset_feet: -0.05
```

根据需要调整这些参数。
