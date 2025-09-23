import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time
import os
import yaml

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

# TODO: 更新README，加上unitree_cpp安装方法
from config_cpp_sdk2 import RobotConfig
from unitree_cpp import UnitreeController, RobotState, SportState

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation_real, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        # robot init
        self.num_dofs = cfg.num_dofs
        self.joint_names = cfg.joint_names
        self.stiffness = cfg.stiffness
        self.damping = cfg.damping
        self.default_pos = cfg.default_pos
        self.dof_idx = cfg.joint2motor_idx

        cfg_unitree = cfg.unitree.to_dict()
        cfg_unitree["num_dofs"] = self.num_dofs
        cfg_unitree["stiffness"] = self.stiffness
        cfg_unitree["damping"] = self.damping
        self.unitree = UnitreeController(cfg_unitree)

        self.msg_type = cfg_unitree["msg_type"]
        self.enable_odometry = cfg_unitree["enable_odometry"]

        self.robot_state: RobotState = None
        self.sport_state: SportState = None

        # Feedback variables, all in radian
        self._joint_positions = np.zeros(self.num_dofs)
        self._joint_velocities = np.zeros(self.num_dofs)
        self._imu_angles = np.zeros(3)
        self._imu_quaternion = np.array([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]
        self._imu_angular_velocity = np.zeros(3)
        self._imu_linear_velocity = np.zeros(3)
        self._base_pos = np.array([0.0, 0.0, 0.9])
        
        self.self_check()
        
        # Robomimic init
        self.remote_controller = RemoteController()
        self.state_cmd = StateAndCmd(self.num_dofs)
        self.policy_output = PolicyOutput(self.num_dofs)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.control_dt = cfg.unitree.control_dt
        
        
        
        
        
        # self.config = config
        # self.num_joints = config.num_joints
        
        
        # self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        # self.low_state = unitree_hg_msg_dds__LowState_()
        # self.mode_pr_ = MotorMode.PR
        # self.mode_machine_ = 0
        # self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        # self.lowcmd_publisher_.Init()
        
        # # inital connection
        # self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        # self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        # self.wait_for_low_state()
        
        # self.init_cmd_hg_cpp(self.low_cmd, self.mode_machine_, self.mode_pr_)
        
        self.policy_output_action = np.zeros(self.num_dofs, dtype=np.float32)
        self.kps = np.zeros(self.num_dofs, dtype=np.float32)
        self.kds = np.zeros(self.num_dofs, dtype=np.float32)
        self.qj = np.zeros(self.num_dofs, dtype=np.float32)
        self.dqj = np.zeros(self.num_dofs, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0,0,-1], dtype=np.float32)
        
        
        
        # self.counter_over_time = 0
        
    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                print("UnitreeCppEnv self check passed!")
                break
        if not self.unitree.self_check():
            print("UnitreeCppEnv self check failed!")
            return False
        return True
    
    # def init_cmd_hg_cpp(self, cmd: LowCmdHG, mode_machine: int, mode_pr: int):
    #     cmd.mode_machine = mode_machine
    #     cmd.mode_pr = mode_pr
    #     size = len(cmd.motor_cmd)
    #     for i in range(size):
    #         cmd.motor_cmd[i].mode = 1
    #         cmd.motor_cmd[i].q = 0
    #         cmd.motor_cmd[i].qd = 0
    #         cmd.motor_cmd[i].kp = 0
    #         cmd.motor_cmd[i].kd = 0
    #         cmd.motor_cmd[i].tau = 0
    
    def update(self):
        self.robot_state = self.unitree.get_robot_state()

        if self.msg_type == "hg":
            self._joint_positions = np.asarray(
                [self.robot_state.motor_state.q[self.dof_idx[i]] for i in range(len(self.dof_idx))]
            )
            self._joint_velocities = np.asarray(
                [self.robot_state.motor_state.dq[self.dof_idx[i]] for i in range(len(self.dof_idx))]
            )
            self._joint_efforts = np.asarray(
                [self.robot_state.motor_state.tau_est[self.dof_idx[i]] for i in range(len(self.dof_idx))]
            )

            quat = np.asarray(self.robot_state.imu_state.quaternion)
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)

            self._imu_quaternion = quat[[1, 2, 3, 0]]
            self._imu_angular_velocity = ang_vel
            self._imu_angles = (
                self.robot_state.imu_state.rpy
            )

        elif self.msg_type == "go":
            raise NotImplementedError("msg_type 'go' not implemented in this example.")
        
        if self.enable_odometry:
            self.sport_state = self.unitree.get_sport_state()
            self._base_pos = np.asarray(self.sport_state.position)
            self._imu_linear_velocity = np.asarray(self.sport_state.velocity)
        
    def step(self, pd_target):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        self.unitree.step(pd_target.tolist())

    def shutdown(self):
        self.unitree.shutdown()

    def set_gains(self, stiffness, damping):
        self.unitree.set_gains(stiffness, damping)
    
    def update_remote_controller(self):
        self.remote_controller.set(self.robot_state.wireless_remote)
        # All low_state is equal to robot_state. The repo makes that
        
        
        
        
        
        
    # def LowStateHgHandler(self, msg: LowStateHG):
    #     self.low_state = msg
    #     self.mode_machine_ = self.low_state.mode_machine
    #     self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    # def wait_for_low_state(self):
    #     while self.low_state.tick == 0:
    #         time.sleep(self.config.control_dt)
    #     print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)
        
    def run(self):
        try:
            # if(self.counter_over_time >= config.error_over_time):
            #     raise ValueError("counter_over_time >= error_over_time")
            
            loop_start_time = time.time()
            
            if self.remote_controller.is_button_pressed(KeyMap.F1):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_1
            if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_2
            # if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
            #     self.state_cmd.skill_cmd = FSMCommand.SKILL_3
            if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.L1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_4  # Beyond Mimic
            
            
            self.update()
            self.update_remote_controller()
            self.state_cmd.vel_cmd[0] =  self.remote_controller.ly
            self.state_cmd.vel_cmd[1] =  self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] =  self.remote_controller.rx * -1

            for i in range(self.num_dofs):
                # self.qj[i] = self.robot_state.motor_state[i].q  # 什么定义了要q迭代而不是motor_state迭代？
                # self.dqj[i] = self.robot_state.motor_state[i].dq  # cpp代码中明明是low_state.motor_state()[i].q() 
                self.qj[i] = self.robot_state.motor_state.q[i]
                self.dqj[i] = self.robot_state.motor_state.dq[i]

            # imu_state quaternion: w, x, y, z
            quat = self.robot_state.imu_state.quaternion
            ang_vel = np.array([self.robot_state.imu_state.gyroscope], dtype=np.float32).squeeze()
            
            gravity_orientation = get_gravity_orientation_real(quat)
            
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = ang_vel.copy()
            self.state_cmd.base_quat = quat
            
            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()
            
            # Build low cmd
            # for i in range(self.num_joints):
            #     self.low_cmd.motor_cmd[i].q = policy_output_action[i]
            #     self.low_cmd.motor_cmd[i].qd = 0
            #     self.low_cmd.motor_cmd[i].kp = kps[i]
            #     self.low_cmd.motor_cmd[i].kd = kds[i]
            #     self.low_cmd.motor_cmd[i].tau = 0
                
            # # send the command
            # # create_damping_cmd(controller.low_cmd) # only for debug
            # self.send_cmd(self.low_cmd)
            self.set_gains(kps, kds)
            self.step(policy_output_action)
            
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            if(delta_time < self.control_dt):
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
            pass
        except ValueError as e:
            print(str(e))
            pass
        
        pass
        
        
if __name__ == "__main__":
    # config = Config()
    # Initialize DDS communication
    # ChannelFactoryInitialize(0, config.net)
    cfg = RobotConfig()
    
    controller = Controller(cfg)
    controller.update()
    
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):  # press SELECT to quit
                break
        except KeyboardInterrupt:
            break
    
    controller.shutdown()
    # create_damping_cmd(controller.low_cmd)
    # controller.send_cmd(controller.low_cmd)
    print("Exit")
    