import numpy as np

def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    return quat_conjugate(q) / np.dot(q, q)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_exp(q):
    v = q[1:]
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])

    v_dir = v / v_norm
    return np.hstack([
        np.cos(v_norm),
        v_dir * np.sin(v_norm)
    ])


def quat_log(q):
    q = quat_normalize(q)
    w = q[0]
    v = q[1:]
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-8:
        return np.zeros(4)

    if q[0] < 0:
        q = -q
    theta = np.arctan2(v_norm, w)
    return np.hstack([
        0.0,
        v / v_norm * theta
    ])


def integrate_gyro(q, omega, dt):
    dq = quat_exp(np.hstack([0.0, 0.5 * omega * dt]))
    q_next = quat_multiply(q, dq)
    return quat_normalize(q_next)


def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])


def quat_to_euler(q):
    q = quat_normalize(q)
    w, x, y, z = q

    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    return np.array([roll, pitch, yaw])

def rotmat_to_rpy(R):
    """
    ZYX convention: yaw-pitch-roll
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw
