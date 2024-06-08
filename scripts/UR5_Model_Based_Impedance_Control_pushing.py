from numpy import linspace
from secrets import choice
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
from manipulator_TDC_sid_wrench import Manipulator
import csv

class ArmSim():
    def __init__(self):
        self.robot = Manipulator([0.8,0.4,0.3],[0.5,0.4,0.8])
        self.ee_pos_data = []
        self.timearray, self.t_step= np.linspace(0,25,2500,retstep=True)
        self.torque_data = np.empty([np.size(self.timearray),len(self.robot.rev)])
        self.alpha_in = [ 0 for i in range(len(self.robot.rev))]
        #print(self.t_step)
        self.h = 1/240
        self.torque_h = np.array([0,0,0,0,0,0])
        self.torque = np.array([0,0,0,0,0,0])        
    def runSim(self):
        self.robot.setInitialState()
        self.robot.turnOffActuators()
        self.robot.turnOffDamping()
        p.enableJointForceTorqueSensor(self.robot.arm,7,enableSensor=1)
        for i,t in zip(range(np.size(self.timearray)),self.timearray):
            p.stepSimulation()
            time.sleep(self.h)
            self.torque_h = self.torque
            self.torque = self.get_torque(t,i)
            self.torque_data[i] = self.torque
            self.ee_pos_data.append(list(self.robot.forces))
            p.setJointMotorControlArray(self.robot.arm,self.robot.rev,controlMode=p.TORQUE_CONTROL,forces=self.torque)
    
    def positionError(self):
        self.positionerror =  self.x_des - np.array(self.robot.eef_pos)
        #print(self.robot.eef_pos)
        return self.positionerror
    
    def velocityError(self):
        self.velocityerror = self.xd_des - np.array(self.robot.eef_vel)
        return self.velocityerror

    def get_torque(self,t,i):
        self.robot.joint_state_info()
        self.robot.getjacobian()
        self.m = self.robot.massMatrix()

        #Impedance Control
        self.M = np.diag(np.array([1e0,1e0,1e0,1e0,1e0,1e0]))
        self.Kp = np.diag([80,80,80,80,80,80])
        self.Kd = np.diag(6*[2*np.sqrt(180)])
    
        if t<=3:
         pos, orn = p.getBasePositionAndOrientation(self.robot.block)
         lin_vel, ang_vel = p.getBaseVelocity(self.robot.block)
         print(pos)
         mat = p.getMatrixFromQuaternion(orn)
         global length
         global point
         length = 0.15

        #Desired Reference
         distance = 0.1  # Distance from the COM to the point
         angle_rad = np.deg2rad(-45)
         point_x = distance * np.cos(angle_rad)
         point_y = distance * np.sin(angle_rad)
         point_z = 0  # Assuming we are rotating in the x-y plane

         # Calculate the coordinates of the point
         point = [pos[0] + point_x, pos[1] + point_y, pos[2] + point_z]
         self.x_des = np.array([pos[0] + point_x, pos[1] + point_y, pos[2] + point_z,0.005,1.45,0.005])
         #self.x_des = np.array([pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7], 0.005,1.45,0.005])
         self.xd_des = np.array([0,0,0,0,0,0])
         self.xdd_des = np.array([0,0,0,0,0,0])
         self.fd = np.array([0,0,0,0,0,0])

         p.addUserDebugLine(pos, [pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6]], [1, 0, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7]], [0, 1, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[2], pos[1] + length*mat[5], pos[2] + length*mat[8]], [0, 0, 1], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, point, [1,0,0], 2, lifeTime = 0.099)

        #p.addUserDebugLine([0,0,0], self.robot.eef_pos[:3],lineColorRGB=[1,1,0],lineWidth=1.5,lifeTime = 0.099)
        #p.addUserDebugLine([0.45,0.439,0.73], [0.45,0.55,0.73],lineColorRGB=[0,0,1],lineWidth=1.5,lifeTime = 0.099)
        #p.addUserDebugLine([0,0,0], self.x_des[:3],lineColorRGB=[1,0,1],lineWidth=1.5,lifeTime = 0.099)
        #p.addUserDebugLine(self.robot.eef_pos[:3], self.x_des[:3],lineColorRGB=[0,0,1],lineWidth=1.5,lifeTime = 0.099)
        
        if t>3:
         pos, orn = p.getBasePositionAndOrientation(self.robot.block)
         lin_vel, ang_vel = p.getBaseVelocity(self.robot.block)
         print(pos)
         mat = p.getMatrixFromQuaternion(orn)
        
        #Desired Reference
         self.x_des = np.array([pos[0],pos[1],pos[2],0.005,1.45,0.005])
         self.xd_des = np.array([0,0,0,0,0,0])
         self.xdd_des = np.array([0,0,0,0,0,0])
         #self.fd = np.array([70*(0.4-pos[0])+80*(0-lin_vel[0]),70*(0.5-pos[1])+80*(0-lin_vel[1]),0,0,0,0])
         self.fd = np.array([0,0,0,0,0,0])

         p.addUserDebugLine(pos, [pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6]], [1, 0, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7]], [0, 1, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[2], pos[1] + length*mat[5], pos[2] + length*mat[8]], [0, 0, 1], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, point, [1,0,0], 2, lifeTime = 0.099)
        """"
        if t>6 and t<=9:
         pos, orn = p.getBasePositionAndOrientation(self.robot.block)
         print(pos)
         mat = p.getMatrixFromQuaternion(orn)

        #Desired Reference
         self.x_des = np.array([pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7],0.005,1.45,0.005])
         self.xd_des = np.array([0,0,0,0,0,0])
         self.xdd_des = np.array([0,0,0,0,0,0])
         self.fd = np.array([0,0,0,0,0,0])

         p.addUserDebugLine(pos, [pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6]], [1, 0, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7]], [0, 1, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[2], pos[1] + length*mat[5], pos[2] + length*mat[8]], [0, 0, 1], 1, lifeTime = 0.099)

        if t>10 and t<=3:
         pos, orn = p.getBasePositionAndOrientation(self.robot.block)
         print(t)
         mat = p.getMatrixFromQuaternion(orn)

        #Desired Reference
         self.x_des = np.array([pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6],0.005,1.45,0.005])
         self.xd_des = np.array([0,0,0,0,0,0])
         self.xdd_des = np.array([0,0,0,0,0,0])
         self.fd = np.array([0,0,0,0,0,0])

         p.addUserDebugLine(pos, [pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6]], [1, 0, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7]], [0, 1, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[2], pos[1] + length*mat[5], pos[2] + length*mat[8]], [0, 0, 1], 1, lifeTime = 0.099)

        if t>13:
         pos, orn = p.getBasePositionAndOrientation(self.robot.block)
         lin_vel, ang_vel = p.getBaseVelocity(self.robot.block)
         print(pos)
         mat = p.getMatrixFromQuaternion(orn)

        #Desired Reference
         self.x_des = np.array([pos[0], pos[1], pos[2],0.005,1.45,0.005])
         self.xd_des = np.array([0,0,0,0,0,0])
         self.xdd_des = np.array([0,0,0,0,0,0])
         self.fd = np.array([120*(0.4-pos[0])+90*(0-lin_vel[0]),0,0,0,0,0])

         p.addUserDebugLine(pos, [pos[0] + length*mat[0], pos[1] + length*mat[3], pos[2] + length*mat[6]], [1, 0, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[1], pos[1] + length*mat[4], pos[2] + length*mat[7]], [0, 1, 0], 1, lifeTime = 0.099)
         p.addUserDebugLine(pos, [pos[0] + length*mat[2], pos[1] + length*mat[5], pos[2] + length*mat[8]], [0, 0, 1], 1, lifeTime = 0.099)
        """
        #self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError() + self.fd)
        self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError()+self.robot.contactForce())
        self.ah = np.linalg.inv(self.robot.analyticjacobian())@(self.alpha_in - self.robot.Jdot@self.robot.omega)
        return self.m@self.ah + self.robot.coriolisVector() + self.robot.gravityVector()-self.robot.analyticjacobian().T@self.robot.contactForce()
       
        

if __name__ == "__main__":
    hl, = plt.plot([], [])
    r1 = ArmSim()
    r1.runSim()


