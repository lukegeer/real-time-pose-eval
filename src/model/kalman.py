import numpy as np

class Kalman:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x


def create_classic_kalman(x_init, y_init):
    x0 = np.array([x_init, y_init, 0, 0])
    P0 = np.eye(4)

    dt = 1.0
    F = np.array([[1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1]])
    B = np.zeros((4, 2))
    H = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0]])
    Q = np.diag([5e-2, 5e-2, 2e-1, 2e-1])
    # Trust measurements more, particularly flow-derived velocity/accel
    R = np.diag([1.0, 1.0])

    kalman = Kalman(F, B, H, Q, R, x0, P0)
    return kalman

def create_kalman_optical_flow_as_measurement(x_init, y_init, dt=1.0):
    x0 = np.array([x_init, y_init, 0, 0])
    P0 = np.eye(4)

    F = np.array([[1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1]])
    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    H = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    Q = np.diag([5e-2, 5e-2, 2e-1, 2e-1])
    # Trust measurements more, particularly flow-derived velocity/accel
    R = np.diag([2e-1, 2e-1, 1.0, 1.0])

    kalman = Kalman(F, B, H, Q, R, x0, P0)
    return kalman

def create_kalman_flow_control_only(x_init, y_init, dt=2.0):
    x0 = np.array([x_init, y_init, 0, 0])
    P0 = np.eye(4)

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    # Control affects velocities directly
    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    # Measure position only
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    Q = np.diag([5e-2, 5e-2, 2e-1, 2e-1])
    # Trust measurements more, particularly flow-derived velocity/accel
    R = np.diag([1.0, 1.0])

    return Kalman(F, B, H, Q, R, x0, P0)


def create_kalman_with_acceleration(x_init, y_init, dt=1.0):
    x0 = np.array([x_init, y_init, 0, 0, 0, 0])
    P0 = np.eye(6)

    dt2 = 0.5 * dt * dt
    F = np.array([
        [1, 0, dt, 0, dt2, 0],
        [0, 1, 0, dt, 0, dt2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    # Control can inject acceleration (first two dims) or jerk (last two dims) if available.
    B = np.array([
        [0, 0, dt2, 0],
        [0, 0, 0, dt2],
        [0, 0, dt, 0],
        [0, 0, 0, dt],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    # Measurement: [x, y, vx, vy, ax, ay]
    H = np.eye(6)
    # Faster-but-stable settings
    Q = np.diag([5e-3, 5e-3, 5e-2, 5e-2, 2e-1, 2e-1])
    R = np.diag([2e-2, 2e-2, 2e-1, 2e-1, 1.0, 1.0])

    kalman = Kalman(F, B, H, Q, R, x0, P0)
    return kalman
