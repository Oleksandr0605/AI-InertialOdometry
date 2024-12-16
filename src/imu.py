import numpy as np
from dataclasses import dataclass
from src.madgwick import Quaternion

@dataclass
class IMUData:
    timestamp: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    orientation: np.ndarray
    sampling_rate: float

    def __post_init__(self):
        self.length = len(self.timestamp)
        if not all(len(x) == self.length for x in [self.gyro, self.acc, self.orientation]):
            raise ValueError("all inputs must have same length")
        
    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(zip(self.gyro, self.acc))

    def process_imu_data(self):
        self.gyro = self.to_quaternion(self.gyro)
        self.acc = self.to_quaternion(self.acc)
        # self.orientation = self.to_quaternion(self.orientation)

    @staticmethod
    def to_quaternion(data: np.ndarray) -> np.ndarray[Quaternion]:
        return np.array([Quaternion(vec) for vec in data])