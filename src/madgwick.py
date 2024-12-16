import numpy as np
from src.quaternion import Quaternion
        
class MadgwickFilter:
    def __init__(self, 
                 initial_quaternion: Quaternion = None, 
                 sampling_rate: int = 200, 
                 beta: float=0.1, 
                 gamma: float=0.5,
                 k: int=1) -> None:
        self.q_est = initial_quaternion if initial_quaternion else Quaternion()
        self.sampling_rate = float(sampling_rate)
        self.delta_t = 1.0 / self.sampling_rate
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.k = int(k)

        self.gravity = Quaternion(np.array([1., 0., 0., 0.])) # normalized gravity vector

    def gyro_orientation_increment(self, q: Quaternion) -> Quaternion:
        """update the quaternion estimate using gyroscope data

        Args:
            gyro (Quaternion): measured 3-axis gyroscope angular velocities as a quaternion, t+1

        Returns:
            Quaternion: updated orientation estimate
        """
        q_est = self.q_est # current quaternion estimate, t
        q_dot = q_est.__mul__(q).__mul__(0.5) # quaterion derivative
        q_est = Quaternion(q_est.__add__(q_dot.__mul__(self.delta_t))) #  an approximation of the quaternion integration step over time interval/sampling rate delta t
        return q_est.normalize()
    
    def acc_orientation_increment(self, q: Quaternion) -> Quaternion:
        """update the quaternion estimate using accelerometer data, gradient descent step

        Args:
            acc (Quaternion): measured 3-axis accelerometer data as a quaternion, t+1

        Returns:
            Quaternion: updated orientation estimate
        """
        q_est = self.q_est
        q_norm = q.normalize()

        for _ in range(self.k):
            J = np.array([
                [-2*q_est.q[1],   2*q_est.q[2],   -2*q_est.q[0],  2*q_est.q[3]],
                [ 2*q_est.q[0],   2*q_est.q[1],    2*q_est.q[3],  2*q_est.q[2]],
                [ 0,           -4*q_est.q[1],   -4*q_est.q[2],  0]
            ])

            f = np.array([
                2 * (q_est.q[1] * q_est.q[3] - q_est.q[0] * q_est.q[2]) - q_norm.q[1],
                2 * (q_est.q[0] * q_est.q[1] + q_est.q[2] * q_est.q[3]) - q_norm.q[2],
                2 * (0.5 - q_est.q[1] * q_est.q[1] - q_est.q[2] * q_est.q[2]) - q_norm.q[3]
            ])

            gradient = J.T @ f

            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < np.finfo(float).eps:
                break

            step = Quaternion(gradient) * (self.beta / gradient_norm)
            q_est += step * (-1)
            q_est = q_est.normalize()

        return q_est
    
    def fusion(self, q_w, q_a) -> Quaternion:
        """fuse gyroscope and accelerometer estimates using weighted sum"""
        q = Quaternion()
        q = q_w.__mul__(1 - self.gamma).__add__(q_a.__mul__(self.gamma))
        return q.normalize()
    
    def update(self, q_w, q_a) -> Quaternion:
        q_w = self.gyro_orientation_increment(q_w)
        q_a = self.acc_orientation_increment(q_a)
        self.q_est = self.fusion(q_w, q_a)
        # return self.q_est.quaternion_euler()
        return self.q_est.q