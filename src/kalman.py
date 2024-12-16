import numpy as np

class KalmanFilter:
    def __init__(self, sampling_rate: int = 200,
                 gyro_noise_density: float = 0.001,
                 gyro_random_walk: float = 0.001, 
                 acc_noise_density: float = 0.001,
                 acc_random_walk: float = 0.001):
        
        self.sampling_rate = float(sampling_rate)
        self.delta_t = 1.0 / self.sampling_rate

        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.acc_noise_density = acc_noise_density
        self.acc_random_walk = acc_random_walk

        gyro_noise_coeff = self.gyro_noise_density ** 2 * self.delta_t
        gyro_walk_coeff = self.gyro_random_walk ** 2 * self.delta_t
        acc_noise_coeff = self.acc_noise_density ** 2 * self.delta_t

        self.state_dim = 6 # state vector [phi, theta, psi, bias_x, bias_y, bias_z]
        self.measurement_dim = 2 # [phi, theta] dont use psi as it is generally inaccurate, in real world cases this value is obtained from camera or compas 

        # state is characterized by:
        self.mu = np.zeros(self.state_dim) # mean
        self.sigma = np.eye(self.state_dim) # cov

        # dynamics noise 
        self.Q = np.eye(self.state_dim)
        # self.Q = np.eye(self.state_dim) * 0.01
        self.Q[0:3, 0:3] = np.eye(3) * gyro_noise_coeff
        self.Q[3:6, 3:6] = np.eye(3) * gyro_walk_coeff

        # measurement noise 
        self.R = np.eye(self.measurement_dim)
        # self.R = np.eye(self.measurement_dim) * 0.01 
        self.R = np.eye(self.measurement_dim) * acc_noise_coeff

        # measurement matrix, mapping from observed state to full state
        self.C = np.zeros((self.measurement_dim, self.state_dim))
        self.C[0,0] = 1.0  # phi
        self.C[1,1] = 1.0  # theta

    def prediction(self, gyro: np.ndarray) -> tuple:
        """ compute the predicted next state using the system dynamics
        input -- gyro measurements
        output -- (predicted mean, predicted var)

        we cosider that bias is not dependent on the attitude 
        """
        
        # A dynamics model -- how state changes from t to t+1
        A = np.eye(self.state_dim)
        A[0:3, 3:6] = -np.eye(3) * self.delta_t

        phi, theta = self.mu[0], self.mu[1]

        # R angle velocities of the IMU to W frame
        R = rotation_matrix(phi, theta)

        # u -- input vector, angle velocities in W frame
        try:
            u = np.linalg.solve(R, gyro)
        except np.linalg.LinAlgError:
            # if singular matrix
            u = np.zeros(3)

        # B mapping of the input vector to state vector 
        B = np.zeros((self.state_dim, 3))
        B[0:3, 0:3] = np.eye(3) * self.delta_t

        self.mu = A @ self.mu + B @ u
        self.sigma = A @ self.sigma @ A.T + self.Q

        return self.mu, self.sigma

    def correction(self, acc) -> tuple:
        """
        input -- acc measurements 
        output -- correction state, combination of dynamic and measurement steps -- sensor fusion (corrected mean, corrected var)
        """

        eps = 1e-7 # avoid div by 0
        phi = np.arctan2(acc[1], np.sqrt(acc[0]**2 + acc[2]**2 + eps))
        theta = np.arctan2(acc[0], np.sqrt(acc[1]**2 + acc[2]**2 + eps))

        # observable state by the sensor 
        z = np.array([phi, theta])

        # K Kalmn gain
        K = self.sigma @ self.C.T @ np.linalg.inv(self.C @ self.sigma @ self.C.T + self.R)

        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.sigma = self.sigma - K @ self.C @ self.sigma

        return self.mu, self.sigma
    
    def fusion(self, gyro, acc):
        self.prediction(gyro)
        self.correction(acc)
        return self.mu, self.sigma

def rotation_matrix(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x


def calculate_trajectory(gyro_data, acc_data, kf: KalmanFilter):
    attitude = np.zeros_like(acc_data)
    velocity = np.zeros_like(acc_data)
    position = np.zeros_like(acc_data)
    gravity = np.array([0, 0, -9.81])

    for i in range(1, len(acc_data)):
        mu, _ = kf.fusion(gyro_data[i-1], acc_data[i-1])

        phi, theta, psi = mu[0], mu[1], mu[2]
        attitude[i] = (phi, theta, psi)

        # rot acc to GF and sub gravity
        R = rotation_matrix(phi, theta, psi)
        acc_GF = R @ acc_data[i] + gravity


        velocity[i] = velocity[i - 1] + acc_GF * kf.delta_t
        position[i] = position[i - 1] + velocity[i] * kf.delta_t

    return position, attitude