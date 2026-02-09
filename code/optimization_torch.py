import numpy as np
import torch

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



def quat_normalize_torch(q):
    return q / torch.norm(q, dim=-1, keepdim=True)


def quat_conjugate_torch(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_inverse_torch(q):
    return quat_conjugate_torch(q) / torch.sum(q * q, dim=-1, keepdim=True)


def quat_multiply_torch(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_exp_torch(q):
    v = q[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)

    v_dir = v / (v_norm + 1e-12)
    w = torch.cos(v_norm)
    xyz = v_dir * torch.sin(v_norm)

    out = torch.cat([w, xyz], dim=-1)
    small = v_norm < 1e-8

    return torch.where(
        small.expand_as(out),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device),
        out,
    )


def quat_log_torch(q):
    q = quat_normalize_torch(q)
    w = q[..., 0:1]
    v = q[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)

    theta = torch.atan2(v_norm, w)
    vec = v / (v_norm + 1e-12) * theta

    out = torch.cat([torch.zeros_like(w), vec], dim=-1)
    small = v_norm < 1e-8

    return torch.where(small.expand_as(out), torch.zeros_like(out), out)


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
    torch.set_num_threads(8)
    
    gyro = torch.tensor(gyro, dtype=torch.float32, device=device)
    accel = torch.tensor(accel, dtype=torch.float32, device=device)

    if isinstance(dt, (int, float)):
        dt_array = torch.full((gyro.shape[0] - 1,), dt, dtype=torch.float32, device=device)
    else:
        dt_array = torch.tensor(dt, dtype=torch.float32, device=device)

    T = gyro.shape[0]
    gravity = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    accel = accel / torch.norm(accel, dim=-1, keepdim=True)

    quats = torch.zeros((T, 4), dtype=torch.float32, device=device)
    quats[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    for t in range(T - 1):
        dq = quat_exp_torch(
            torch.cat([
                torch.tensor([0.0], device=device),
                0.5 * gyro[t] * dt_array[t]
            ])
        )
        quats[t + 1] = quat_multiply_torch(quats[t], dq)

    quats = quat_normalize_torch(quats)
    quats = quats.clone().detach()
    quats.requires_grad_(True)

    for it in range(num_iters):
        with torch.no_grad():
            quats[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        cost = 0.0

        for t in range(T - 1):
            dq = quat_exp_torch(
                torch.cat([
                    torch.tensor([0.0], device=device),
                    0.5 * gyro[t] * dt_array[t]
                ])
            )
            q_pred = quat_multiply_torch(quats[t], dq)
            q_err = quat_multiply_torch(quat_inverse_torch(quats[t + 1]), q_pred)
            r_g = 2.0 * quat_log_torch(q_err)[1:]
            cost = cost + 0.5 * w_g * torch.dot(r_g, r_g)

        for t in range(1, T):
            q_inv = quat_inverse_torch(quats[t])
            g_body = quat_multiply_torch(
                quat_multiply_torch(q_inv, torch.cat([torch.tensor([0.0], device=device), gravity])),
                quats[t]
            )[1:]
            r_a = accel[t - 1] - g_body
            cost = cost + 0.5 * w_a * torch.dot(r_a, r_a)

        cost.backward()

        with torch.no_grad():
            for t in range(1, T):
                delta = -step_size * quats.grad[t][1:]
                dq = quat_exp_torch(torch.cat([torch.tensor([0.0], device=device), delta]))
                quats[t] = quat_normalize_torch(quat_multiply_torch(quats[t], dq))
            quats.grad.zero_()

        if it % 1 == 0: print(f"Iter {it:05d} | cost = {cost.item():.6f}")

    return quats.detach().cpu().numpy()