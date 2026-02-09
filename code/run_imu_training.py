#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

from optimization import projected_gradient_descent
from quaternion_utils import quat_to_rotmat, rotmat_to_rpy, integrate_gyro


def read_data(fname):
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')


def extract_vicon_rot(data):
    if isinstance(data, dict) and 'rots' in data:
        return data['rots'], data.get('ts', None)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        item = data.item()
        return item['rots'], item.get('ts', None)
    else:
        raise ValueError("Unsupported VICON format")


def integrate_gyro_trajectory(gyro_phys, dt):
    N = gyro_phys.shape[1]
    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    q_history = [q_init]
    
    for i in range(N - 1):
        q_next = integrate_gyro(q_history[-1], gyro_phys[:, i], dt[i])
        q_history.append(q_next)
    
    return np.array(q_history)


def plot_orientation_comparison(t_imu, quats_init, quats_opt, vicon_rots, vicon_ts, dataset):
    vicon_ts_rel = vicon_ts - vicon_ts[0]
    
    t_start = max(t_imu[0], vicon_ts_rel[0])
    t_end = min(t_imu[-1], vicon_ts_rel[-1])
    
    mask = (t_imu >= t_start) & (t_imu <= t_end)
    t_common = t_imu[mask]
    quats_init_common = quats_init[mask]
    quats_opt_common = quats_opt[mask]
    
    vicon_rots_interp = np.zeros((3, 3, len(t_common)))
    for i in range(3):
        for j in range(3):
            f = interp1d(vicon_ts_rel, vicon_rots[i, j, :], 
                        kind='linear', bounds_error=False, fill_value='extrapolate')
            vicon_rots_interp[i, j, :] = f(t_common)
    
    T_common = len(t_common)
    
    rpy_init = np.zeros((T_common, 3))
    rpy_opt = np.zeros((T_common, 3))
    rpy_vic = np.zeros((T_common, 3))
    
    for t_idx in range(T_common):
        R_init = quat_to_rotmat(quats_init_common[t_idx])
        R_opt = quat_to_rotmat(quats_opt_common[t_idx])
        R_vic = vicon_rots_interp[:, :, t_idx]
        
        rpy_init[t_idx] = np.array(rotmat_to_rpy(R_init))
        rpy_opt[t_idx] = np.array(rotmat_to_rpy(R_opt))
        rpy_vic[t_idx] = np.array(rotmat_to_rpy(R_vic))
    
    for i in range(3):
        rpy_init[:, i] = np.unwrap(rpy_init[:, i])
        rpy_opt[:, i] = np.unwrap(rpy_opt[:, i])
        rpy_vic[:, i] = np.unwrap(rpy_vic[:, i])
    
    errors_init = rpy_init - rpy_vic
    errors_opt = rpy_opt - rpy_vic
    
    rmse_init = np.sqrt(np.mean(errors_init**2, axis=0))
    rmse_opt = np.sqrt(np.mean(errors_opt**2, axis=0))
    
    print(f"\n=== ORIENTATION ERRORS ===")
    print(f"Initial (Gyro Integration):")
    print(f"  RMSE (deg): Roll={np.rad2deg(rmse_init[0]):.3f}, Pitch={np.rad2deg(rmse_init[1]):.3f}, Yaw={np.rad2deg(rmse_init[2]):.3f}")
    print(f"Optimized (PGD):")
    print(f"  RMSE (deg): Roll={np.rad2deg(rmse_opt[0]):.3f}, Pitch={np.rad2deg(rmse_opt[1]):.3f}, Yaw={np.rad2deg(rmse_opt[2]):.3f}")
    
    labels = ["Roll", "Pitch", "Yaw"]
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    for i in range(3):
        axs[i].plot(t_common, rpy_vic[:, i], 'b--', label='VICON Ground Truth', linewidth=1.75)
        axs[i].plot(t_common, rpy_init[:, i], 'g-', label='Initial (Gyro Integration)', linewidth=1, alpha=0.7)
        axs[i].plot(t_common, rpy_opt[:, i], 'r-', label='Optimized (PGD)', linewidth=1.75)
        
        axs[i].set_ylabel(f'{labels[i]} (rad)', fontsize=11)
        axs[i].set_title(f'{labels[i]} - RMSE Init: {np.rad2deg(rmse_init[i]):.2f}°, Opt: {np.rad2deg(rmse_opt[i]):.2f}°', 
                        fontsize=11, fontweight='bold')
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='upper right')
    
    axs[2].set_xlabel('Time (s)', fontsize=11)
    plt.suptitle(f'Orientation Tracking Comparison - Dataset {dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    orientation_file = f"orientation_comparison_dataset{dataset}.png"
    plt.savefig(orientation_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {orientation_file}")
    plt.show()


def main():
    dataset = 7
    imu_file = f"../data/trainset/imu/imuRaw{dataset}.p"
    vicon_file = f"../data/trainset/vicon/viconRot{dataset}.p"
    
    print("=" * 70)
    print(f"ORIENTATION TRACKING - Dataset {dataset}")
    print("=" * 70)
    
    imud = read_data(imu_file)
    vicd = read_data(vicon_file)
    
    t_imu = imud[0, :]
    acc_raw = imud[1:4, :]
    gyro_raw = imud[4:7, :]
    vicon_rots, vicon_ts = extract_vicon_rot(vicd)
    
    t = t_imu - t_imu[0]
    dt = np.diff(t)
    
    V_REF = 3300.0
    ADC_RES = 1023.0
    SENS_ACCEL = 330.0
    SENS_GYRO_RAD = 3.33 * (180.0 / np.pi)
    
    scale_accel = V_REF / ADC_RES / SENS_ACCEL
    scale_gyro = V_REF / ADC_RES / SENS_GYRO_RAD
    
    g_raw = np.array([[0.0], [0.0], [1.0 / scale_accel]])
    acc_bias = np.mean(acc_raw[:, :50], axis=1, keepdims=True) - g_raw
    gyro_bias = np.mean(gyro_raw[:, :50], axis=1, keepdims=True)
    
    acc_phys = (acc_raw - acc_bias) * scale_accel
    gyro_phys = (gyro_raw - gyro_bias) * scale_gyro
    
    print("\n=== INITIAL GYRO INTEGRATION ===")
    quats_init = integrate_gyro_trajectory(gyro_phys, dt)
    print(f"Integrated {len(quats_init)} quaternions")
    
    print("\n=== OPTIMIZATION ===")
    quats_opt = projected_gradient_descent(
        gyro_phys.T, acc_phys.T, dt,
        num_iters=15, step_size=1e-1, w_g=0.5, w_a=0.5,
    )
    
    plot_orientation_comparison(t, quats_init, quats_opt, vicon_rots, vicon_ts.flatten(), dataset)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()