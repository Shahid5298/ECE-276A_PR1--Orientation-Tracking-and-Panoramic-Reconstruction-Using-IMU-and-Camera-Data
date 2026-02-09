import numpy as np
from quaternion_utils import (
    quat_multiply,
    quat_inverse,
    quat_log,
    quat_exp,
    quat_normalize,
)

GRAVITY = np.array([0.0, 0.0, 1.0])


def gyro_residual(q_t, q_tp1, omega_t, dt):
    dq = quat_exp(np.hstack(([0.0], 0.5 * omega_t * dt)))
    q_pred = quat_multiply(q_t, dq)
    q_err = quat_multiply(quat_inverse(q_tp1), q_pred)
    return 2.0 * quat_log(q_err)[1:]


def accel_residual(q_t, accel_t):
    q_inv = quat_inverse(q_t)
    g_body = quat_multiply(
        quat_multiply(q_inv, np.hstack(([0.0], GRAVITY))),
        q_t,
    )[1:]
    return accel_t - g_body


def total_cost(quats, gyro, accel, dt, w_g, w_a):
    cost = 0.0
    T = quats.shape[0]

    for t in range(T - 1):
        r_g = gyro_residual(quats[t], quats[t + 1], gyro[t], dt)
        cost += w_g * np.dot(r_g, r_g)

    for t in range(1, T):
        r_a = accel_residual(quats[t], accel[t - 1])
        cost += w_a * np.dot(r_a, r_a)

    return cost


def projected_gradient_descent(
    gyro,
    accel,
    dt,
    num_iters,
    step_size,
    w_g=0.5,
    w_a=0.5,
    device="cpu"
):
    gyro = np.array(gyro, dtype=np.float32)
    accel = np.array(accel, dtype=np.float32)

    if isinstance(dt, (int, float)):
        dt_array = np.full((gyro.shape[0] - 1,), dt, dtype=np.float32)
    else:
        dt_array = np.array(dt, dtype=np.float32)

    T = gyro.shape[0]
    gravity = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    accel = accel / np.linalg.norm(accel, axis=-1, keepdims=True)

    quats = np.zeros((T, 4), dtype=np.float32)
    quats[0] = np.array([1.0, 0.0, 0.0, 0.0])

    for t in range(T - 1):
        dq = quat_exp(
            np.hstack([
                [0.0],
                0.5 * gyro[t] * dt_array[t]
            ])
        )
        quats[t + 1] = quat_multiply(quats[t], dq)

    quats = np.array([quat_normalize(q) for q in quats])

    for it in range(num_iters):
        quats[0] = np.array([1.0, 0.0, 0.0, 0.0])

        grads = np.zeros((T, 3), dtype=np.float32)
        cost = 0.0

        for t in range(T - 1):
            dq = quat_exp(
                np.hstack([
                    [0.0],
                    0.5 * gyro[t] * dt_array[t]
                ])
            )
            q_pred = quat_multiply(quats[t], dq)
            q_err = quat_multiply(quat_inverse(quats[t + 1]), q_pred)
            r_g = 2.0 * quat_log(q_err)[1:]
            cost += 0.5 * w_g * np.dot(r_g, r_g)


            eps = 1e-6
            for k in range(3):
                delta = np.zeros(3)
                delta[k] = eps
                dq_eps = quat_exp(np.hstack(([0.0], delta)))
                q_tp1_pert = quat_normalize(quat_multiply(quats[t + 1], dq_eps))
                
                q_err_pert = quat_multiply(quat_inverse(q_tp1_pert), q_pred)
                r_g_pert = 2.0 * quat_log(q_err_pert)[1:]
                
                grad_k = (r_g_pert - r_g) / eps
                grads[t + 1, k] += w_g * np.dot(r_g, grad_k)

        for t in range(1, T):
            q_inv = quat_inverse(quats[t])
            g_body = quat_multiply(
                quat_multiply(q_inv, np.hstack(([0.0], gravity))),
                quats[t]
            )[1:]
            r_a = accel[t - 1] - g_body
            cost += 0.5 * w_a * np.dot(r_a, r_a)

            eps = 1e-6
            for k in range(3):
                delta = np.zeros(3)
                delta[k] = eps
                dq_eps = quat_exp(np.hstack(([0.0], delta)))
                q_pert = quat_normalize(quat_multiply(quats[t], dq_eps))
                
                q_inv_pert = quat_inverse(q_pert)
                g_body_pert = quat_multiply(
                    quat_multiply(q_inv_pert, np.hstack(([0.0], gravity))),
                    q_pert
                )[1:]
                r_a_pert = accel[t - 1] - g_body_pert
                
                grad_k = (r_a_pert - r_a) / eps
                grads[t, k] += w_a * np.dot(r_a, grad_k)

        for t in range(1, T):
            delta = -step_size * grads[t]
            dq = quat_exp(np.hstack(([0.0], delta)))
            quats[t] = quat_normalize(quat_multiply(quats[t], dq))

        if it % 1 == 0:
            print(f"Iter {it:05d} | cost = {cost:.6f}")

    return quats