import numpy as np


PARAMETER_BYTES = 4_161_296


def flops(image_size, batch):
    s, b = np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)
    return b * (17_714 * s**2 + 313_700)


def memory(image_size, batch):
    """Peak tensor bytes, including MaxPool"""
    s, b = np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)
    return PARAMETER_BYTES + 68 * b * s**2


def bytes_moved(image_size, batch):
    s, b = np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)
    return PARAMETER_BYTES + 380 * b * s**2 + 8_592 * b


def latency(image_size, batch, theta):
    t0_ms, p_tflops, bw_gbs = theta
    compute = flops(image_size, batch) / (p_tflops * 1e12)
    transfer = bytes_moved(image_size, batch) / (bw_gbs * 1e9)
    return t0_ms / 1000 + np.maximum(compute, transfer)


def energy(image_size, batch, theta_energy):
    e0, e_flops, e_bytes = theta_energy
    return (
        e0
        + e_flops * flops(image_size, batch) / 1e12
        + e_bytes * bytes_moved(image_size, batch) / 1e9
    )
