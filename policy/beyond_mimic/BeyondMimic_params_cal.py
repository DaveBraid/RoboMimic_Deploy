#!/usr/bin/env python3

import numpy as np

# --------------------------------------------------------------------------
# 步骤0: 选定生成参数的顺序
OUTPUT_IN_MJ_ORDER = True  # True: 直接输出按 MuJoCo 关节顺序排列的参数, False: 输出 Lab 顺序
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 步骤 1: 复制 S3_CYLINDER_CFG 中的所有物理常量
# (如果调整原文件，需在此处同步修改1)
# --------------------------------------------------------------------------
ARMATURE_5020 = 0.01
ARMATURE_7520_14 = 0.02
ARMATURE_7520_22 = 0.02
ARMATURE_4010 = 0.02

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


# --------------------------------------------------------------------------
# 步骤 2: 将 S3_CYLINDER_CFG 转录为标准 Python 字典
# (这里使用变量名，因此计算会自动更新)
# (如果调整原文件，需在此处同步修改2)
# --------------------------------------------------------------------------
S3_CONFIG_DATA = {
    "init_state": {
        "joint_pos": {
            ".*_hip_pitch_joint": -0.22,
            ".*_knee_joint": 0.56,
            ".*_foot_pitch_joint": -0.363,
            # "left_shoulder_roll_joint": 1.57,
            # "right_shoulder_roll_joint": -1.57,
        },
    },
    "actuators": {
        "legs": {
            "joint_names_expr": [
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            "effort_limit_sim": {
                ".*_hip_yaw_joint": 300.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
            },
            "stiffness": {
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            "damping": {
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
        },
        "feet": {
            "effort_limit_sim": 300.0,  # 这是一个 float, 会应用到所有关节
            "joint_names_expr": [".*_foot_pitch_joint", ".*_foot_roll_joint"],
            "stiffness": 2.0 * STIFFNESS_5020, # 这是一个 float
            "damping": 2.0 * DAMPING_5020,   # 这是一个 float
        },
        "arms": {
            "joint_names_expr": [
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_hand_joint",
            ],
            "effort_limit_sim": {
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_hand_joint": 25.0,
            },
            "stiffness": {
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_hand_joint": STIFFNESS_4010,
            },
            "damping": {
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_hand_joint": DAMPING_4010,
            },
        },
    },
}

# --------------------------------------------------------------------------
# 步骤 3: 定义关节顺序 (Lab 顺序 和 MuJoCo 顺序)
# --------------------------------------------------------------------------

# "Lab 顺序" (由 ? 定义)  # TODO
lab_order_names = [
    'left_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_hip_roll_joint', 'right_shoulder_pitch_joint',
    'left_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_hip_yaw_joint', 'right_shoulder_roll_joint', 
    'left_hip_pitch_joint', 'left_shoulder_yaw_joint', 'right_hip_pitch_joint', 'right_shoulder_yaw_joint', 
    'left_knee_joint', 'left_elbow_joint', 'right_knee_joint', 'right_elbow_joint', 'left_foot_pitch_joint',
    'left_hand_joint', 'right_foot_pitch_joint', 'right_hand_joint', 'left_foot_roll_joint', 'right_foot_roll_joint'
 ]

# "MuJoCo 顺序" (由 S3_22dof.xml 的运动学树遍历决定)
mj_order_names = [
    "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint", "left_knee_joint", "left_foot_pitch_joint", "left_foot_roll_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint", "right_knee_joint", "right_foot_pitch_joint", "right_foot_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_hand_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_hand_joint",
]
# mj_order_names = [
#     "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_hand_joint",
#     "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_hand_joint",
#     "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint", "left_knee_joint", "left_foot_pitch_joint", "left_foot_roll_joint",
#     "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint", "right_knee_joint", "right_foot_pitch_joint", "right_foot_roll_joint",
# ]

# --------------------------------------------------------------------------
# 步骤 4: 辅助函数和主计算逻辑
# --------------------------------------------------------------------------

def expand_joint_expr(expr):
    """
    展开关节名表达式。
    ".*_joint_name" -> ["left_joint_name", "right_joint_name"]
    "joint_name" -> ["joint_name"]
    """
    if expr.startswith(".*_"):
        base_name = expr[3:]
        return [f"left_{base_name}", f"right_{base_name}"]
    return [expr]

def get_property_map(config_dict, prop_name):
    """
    从配置字典中提取特定属性 (如 'stiffness', 'damping') 
    并将其展开为 (joint_name -> value) 的映射。
    """
    prop_map = {}
    for group_name, group_cfg in config_dict.items():
        prop_source = group_cfg.get(prop_name)
        if prop_source is None:
            continue
            
        joint_exprs = group_cfg["joint_names_expr"]
        
        if isinstance(prop_source, (float, int)):
            # 属性是单个值 (如 feet.stiffness)
            for expr in joint_exprs:
                for joint_name in expand_joint_expr(expr):
                    prop_map[joint_name] = prop_source
        
        elif isinstance(prop_source, dict):
            # 属性是一个字典 (如 legs.stiffness)
            for expr in joint_exprs:
                # 在字典源中找到匹配的 key
                matching_key = None
                if expr in prop_source:
                    matching_key = expr
                elif expr.replace(".*_", "left_") in prop_source:
                     # 处理非.*_的特定表达式 (虽然S3中没有，但保持鲁棒)
                    matching_key = expr.replace(".*_", "left_")
                elif expr.replace(".*_", "right_") in prop_source:
                    matching_key = expr.replace(".*_", "right_")
                
                # S3 配置中，key 总是与 joint_exprs 中的表达式匹配
                if matching_key is None:
                     matching_key = expr

                if matching_key in prop_source:
                    value = prop_source[matching_key]
                    for joint_name in expand_joint_expr(expr):
                        prop_map[joint_name] = value
                else:
                    # 处理特定关节 (如 left_shoulder_roll_joint)
                    if expr in prop_source:
                         prop_map[expr] = prop_source[expr]

    return prop_map

def get_default_angle_map(config_dict):
    """
    从 init_state.joint_pos 展开默认角度映射。
    """
    angle_map = {}
    for expr, angle in config_dict.items():
        for joint_name in expand_joint_expr(expr):
            angle_map[joint_name] = angle
    return angle_map

def print_yaml_array(name, data, order_label, precision=4, per_row=6):
    """辅助函数，用于按照指定顺序漂亮地打印 YAML 数组"""
    print(f"# {name} (按 {order_label} 顺序)")
    print(f"{name.lower()}: [", end="")
    
    for i, val in enumerate(data):
        if i % per_row == 0:
            print("\n  ", end="")
        
        if isinstance(val, (int, np.integer)):
            formatted_val = f"{val}"
        else:
            formatted_val = f"{val:.{precision}f}"
            
        print(f"{formatted_val}, ", end="")
    
    print("\n]")
    print("#", "-" * 60) # 分隔符

def main():
    """
    主执行函数：
    1. 展开所有属性映射
    2. 按选定顺序构建数组
    3. 计算 mj2lab 映射
    4. 打印所有结果
    """
    
    # --- 1. 展开所有属性映射 ---
    actuator_config = S3_CONFIG_DATA["actuators"]
    kp_map = get_property_map(actuator_config, "stiffness")
    kd_map = get_property_map(actuator_config, "damping")
    tau_map = get_property_map(actuator_config, "effort_limit_sim")
    
    angle_config = S3_CONFIG_DATA["init_state"]["joint_pos"]
    angle_map = get_default_angle_map(angle_config)
    
    # --- 2. 按选择的顺序构建数组 ---
    joint_order = mj_order_names if OUTPUT_IN_MJ_ORDER else lab_order_names
    order_label = "MuJoCo" if OUTPUT_IN_MJ_ORDER else "Lab"
    name_suffix = "mj" if OUTPUT_IN_MJ_ORDER else "lab"

    kp_values = []
    kd_values = []
    tau_limit = []
    action_scale_values = []
    default_angles = []
    
    for joint_name in joint_order:
        if joint_name not in kp_map:
            print(f"** 错误: 关节 '{joint_name}' 在 kp_map (stiffness) 中未找到")
            continue
        if joint_name not in kd_map:
            print(f"** 错误: 关节 '{joint_name}' 在 kd_map (damping) 中未找到")
            continue
        if joint_name not in tau_map:
            print(f"** 错误: 关节 '{joint_name}' 在 tau_map (effort_limit_sim) 中未找到")
            continue
            
        kp = kp_map[joint_name]
        kd = kd_map[joint_name]
        tau = tau_map[joint_name]

        kp_values.append(kp)
        kd_values.append(kd)
        tau_limit.append(tau)
        action_scale_values.append(tau / kp * 0.25) # S3_ACTION_SCALE = 0.25 * e[n] / s[n]
        default_angles.append(angle_map.get(joint_name, 0.0)) # 未指定则默认为 0.0
        
    # --- 3. 计算 mj2lab 映射 ---
    lab_to_index_map = {name: idx for idx, name in enumerate(mj_order_names)}
    mj2lab = [lab_to_index_map[name] for name in lab_order_names]
    
    # --- 4. 打印所有结果 ---
    print("#", "=" * 60)
    print("# S3 (22-DOF) BeyondMimic YAML 配置值 (由params cal脚本自动计算)")
    print("#", "=" * 60)

    print(f"# 当前输出顺序: {order_label}")
    print_yaml_array(f"kp_{name_suffix}", kp_values, order_label, precision=3)
    print_yaml_array(f"kd_{name_suffix}", kd_values, order_label, precision=3)
    print_yaml_array(f"tau_limit_{name_suffix}", tau_limit, order_label, precision=1)
    print_yaml_array(f"action_scale_{name_suffix}", action_scale_values, order_label, precision=4)
    print_yaml_array(f"default_angles_{name_suffix}", default_angles, order_label, precision=3)

    print("# MuJoCo 索引 -> Lab 索引 映射")
    print("mj2lab: [", end="")
    for i, val in enumerate(mj2lab):
        if i % 12 == 0:
            print("\n  ", end="")
        print(f"{val}, ", end="")
    print("\n]")
    print("#", "-" * 60)

if __name__ == "__main__":
    main()