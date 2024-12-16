import numpy as np
from typing import Union, List, Tuple

class Quaternion:
    def __init__(self, x: Union[np.ndarray, List, float, None] = None) -> None:
        if x is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif isinstance(x, Quaternion):
            self.q = x.q.copy()
        elif isinstance(x, (list, np.ndarray)):
            arr = np.asarray(x, dtype=np.float64)
            if arr.ndim == 1:
                if arr.size == 3:
                    # self.q = self.euler_quaternion(arr)
                    self.q = np.array([0.0, arr[0], arr[1], arr[2]])
                elif arr.size == 4:
                    self.q = arr.copy()
                else:
                    raise ValueError("input array must have size 3 or 4")
            elif arr.ndim == 2 and arr.shape[1] == 3:
                self.q = np.hstack([np.zeros((arr.shape[0], 1)), arr]) # todo: for qaut
                # self.q = np.array([self.euler_quaternion(a) for a in arr])
            else:
                raise ValueError("invalid input dimensions")

    def magnitude(self) -> float:
        return np.sqrt(np.sum(self.q**2))
    
    def normalize(self):
        mag = self.magnitude()
        if mag < np.finfo(float).eps:
            return self
        return Quaternion(self.q / mag)
    
    def conjugate(self):
        q_conj = Quaternion()
        q_conj.q = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        return q_conj

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.q + other.q)
        
    def __mul__(self, other):
        q = self.q
        if isinstance(other, Quaternion):
            other_q = other.q
            result = Quaternion()
            result.q = np.array([
                q[0]*other_q[0] - q[1]*other_q[1] - q[2]*other_q[2] - q[3]*other_q[3],
                q[0]*other_q[1] + q[1]*other_q[0] + q[2]*other_q[3] - q[3]*other_q[2],
                q[0]*other_q[2] - q[1]*other_q[3] + q[2]*other_q[0] + q[3]*other_q[1],
                q[0]*other_q[3] + q[1]*other_q[2] - q[2]*other_q[1] + q[3]*other_q[0]
            ])
            return result
        elif isinstance(other, (int, float)):
            return Quaternion(self.q * other)
        
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Quaternion(self.q / scalar)

    def euler_quaternion(self, angle):
        """euler to quaternion
        ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        """
        angle = np.radians(angle)

        cos_x = np.cos(angle[0] * 0.5)
        sin_x = np.sin(angle[0] * 0.5)
        cos_y = np.cos(angle[1] * 0.5)
        sin_y = np.sin(angle[1] * 0.5)
        cos_z = np.cos(angle[2] * 0.5)
        sin_z = np.sin(angle[2] * 0.5)
        
        qw = cos_x * cos_y * cos_z + sin_x * sin_y * sin_z
        qx = sin_x * cos_y * cos_z - cos_x * sin_y * sin_z
        qy = cos_x * sin_y * cos_z + sin_x * cos_y * sin_z
        qz = cos_x * cos_y * sin_z - sin_x * sin_y * cos_z

        return np.array([qw, qx, qy, qz], dtype=np.float64)

    def quaternion_euler(self) -> Tuple[float, float, float]:
        q0, q1, q2, q3 = self.q
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return roll, pitch, yaw
        
    def __str__(self):
        return f"{self.q.tolist()}"
    
    def __repr__(self) -> str:
        return f"{self.q.tolist()}"
    
    def quaternion_to_rotation_matrix(self):
        self = self.normalize()
        w, x, y, z = self.q
        R = np.array([
            [2*(w**2 + x**2) - 1, 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 2*(w**2 + y**2) - 1, 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 2*(w**2 + z**2) - 1]
        ])
        return R