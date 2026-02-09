#!/usr/bin/env python3
import sys
import pickle
import numpy as np
from pathlib import Path

from imu_calibration import (
    calibrate_imu,
    find_static_interval,
    calibrate_gyro_bias,
    calibrate_accel_bias,
)

def read_data(fname):
    with open(fname, "rb") as f:
        if sys.version_info[0] < 3:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding="latin1")

def extract_imu_dict(data):
    if isinstance(data, dict):
        if "vals" in data and "ts" in data:
            return data
        else:
            raise ValueError(f"Dict missing keys {data.keys()}")

    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            item = data.item() if data.shape == () else data[0]
            if isinstance(item, dict) and "vals" in item and "ts" in item:
                return item
            raise ValueError("Object array contains wrong format")

        elif data.dtype.names is not None:
            if "vals" in data.dtype.names and "ts" in data.dtype.names:
                return {
                    "vals": data["vals"][0] if data.shape != () else data["vals"],
                    "ts": data["ts"][0] if data.shape != () else data["ts"],
                }
            else:
                raise ValueError("Structured array does not contain vals/ts")

        elif len(data.shape) == 2 and data.shape[0] == 7:
            return {"vals": data[0:6, :], "ts": data[6:7, :]}

        elif len(data.shape) >= 1 and data.shape[0] == 2:
            return {"vals": data[0], "ts": data[1]}

        raise ValueError(f"Cannot parse array shape {data.shape}")

    elif isinstance(data, list) and len(data) >= 2:
        return {"vals": np.array(data[0]), "ts": np.array(data[1])}

    else:
        raise ValueError(f"Unsupported IMU data type {type(data)}")

def display_calibration_params(gyro_bias, accel_bias):
    print("\nAccelerometer bias (raw ADC):", accel_bias.flatten())
    print("Gyroscope bias (raw ADC):", gyro_bias.flatten())

def verify_calibration(accel_raw, gyro_raw, accel_cal, gyro_cal, static_idx):
    accel_mean_raw = np.mean(accel_raw[:, static_idx], axis=1)
    gyro_mean_raw = np.mean(gyro_raw[:, static_idx], axis=1)

    accel_mean_cal = np.mean(accel_cal[:, static_idx], axis=1)
    gyro_mean_cal = np.mean(gyro_cal[:, static_idx], axis=1)

    gyro_drift_before = np.linalg.norm(gyro_mean_raw)
    gyro_drift_after = np.linalg.norm(gyro_mean_cal)

    accel_mag_after = np.mean(np.linalg.norm(accel_cal[:, static_idx], axis=0))

    print("\nGyro drift before calibration:", gyro_drift_before)
    print("Gyro drift after calibration:", gyro_drift_after)
    print("Accel magnitude after calibration (should be ~1g):", accel_mag_after)

    return gyro_drift_after < gyro_drift_before

def main():
    Vref = 3300.0

    ACCEL_SENS = 330.0  
    GYRO_SENS = 3.33 * (180.0 / np.pi) 

    accel_scale = Vref / (1023.0 * ACCEL_SENS)
    gyro_scale = Vref / (1023.0 * GYRO_SENS)

    STATIC_TIME = 2.0

    dataset_num = 1
    ifile = f"../data/trainset/imu/imuRaw{dataset_num}.p"

    if not Path(ifile).exists():
        print(f"File not found: {ifile}")
        return

    data_raw = read_data(ifile)
    imu_data = extract_imu_dict(data_raw)

    vals = imu_data["vals"]
    ts = imu_data["ts"].flatten()

    accel_raw = vals[0:3, :]
    gyro_raw = vals[3:6, :]

    static_idx = find_static_interval(ts, static_time=STATIC_TIME)

    gyro_bias = calibrate_gyro_bias(gyro_raw * gyro_scale, static_idx)
    accel_bias = calibrate_accel_bias(accel_raw * accel_scale, static_idx)

    accel_cal, gyro_cal, ts_cal = calibrate_imu(
        imu_data, accel_scale, gyro_scale
    )

    calibration_good = verify_calibration(
        accel_raw, gyro_raw, accel_cal, gyro_cal, static_idx
    )

    print("\nCalibration result:", "GOOD" if calibration_good else "Check parameters")

if __name__ == "__main__":
    main()
