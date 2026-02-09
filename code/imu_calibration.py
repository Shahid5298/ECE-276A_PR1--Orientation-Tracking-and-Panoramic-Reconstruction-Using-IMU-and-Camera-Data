import numpy as np
import matplotlib.pyplot as plt

def find_static_interval(imu_ts, static_time=2.0):
    """
    Identify indices corresponding to the first static_time seconds.
    """
    t0 = imu_ts[0]
    idx = np.where(imu_ts <= t0 + static_time)[0]
    return idx


def calibrate_gyro_bias(gyro, static_idx):
    """
    Gyro bias estimated as mean angular velocity during static interval.
    """
    bias = np.mean(gyro[:, static_idx], axis=1, keepdims=True)
    return bias


def calibrate_accel_bias(accel, static_idx):
    """
    Accelerometer bias estimated so that mean accel during static
    equals [0, 0, 1] in g units.
    """
    accel_mean = np.mean(accel[:, static_idx], axis=1, keepdims=True)
    g_true = np.array([[0.0], [0.0], [1.0]])
    bias = accel_mean - g_true
    return bias


def calibrate_imu(imu_data, accel_scale, gyro_scale):

    vals = imu_data['vals']
    ts = imu_data['ts'].flatten()

    accel_raw = vals[0:3, :]
    gyro_raw  = vals[3:6, :]

    static_idx = find_static_interval(ts, static_time=2.0)

    gyro_bias  = calibrate_gyro_bias(gyro_raw, static_idx)
    accel_bias = calibrate_accel_bias(accel_raw * accel_scale, static_idx)

    accel_cal = (accel_raw * accel_scale) - accel_bias
    gyro_cal  = (gyro_raw  - gyro_bias)  * gyro_scale

    return accel_cal, gyro_cal, ts