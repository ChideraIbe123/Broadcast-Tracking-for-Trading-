"""Compare motion models on synthetic trajectories: sharp cut, curve, stop, straight."""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from hoops.motion import IMM, KalmanCA, KalmanCV, ParticleFilter  # noqa: E402


def sharp_cut(n=60):
    a = np.array([10.0, 20.0]) + np.arange(30)[:, None] * np.array([3.0, 0.0])
    b = a[-1] + np.arange(1, 31)[:, None] * np.array([0.0, 3.0])
    return np.vstack([a, b])


def curve(n=60, r=25.0, v=3.0):
    th = np.arange(n) * v / r
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def stop(n=60):
    a = np.array([10.0, 20.0]) + np.arange(35)[:, None] * np.array([3.0, 0.0])
    return np.vstack([a, np.repeat(a[-1:], 25, 0)])


def straight(n=60):
    return np.array([10.0, 20.0]) + np.arange(n)[:, None] * np.array([2.0, -1.0])


def run(model_fn, truth, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    zs = truth + rng.normal(0, noise, truth.shape)
    m = model_fn(zs[0])
    err = []
    for i, z in enumerate(zs[1:], 1):
        p = m.predict()
        err.append(np.linalg.norm(p - truth[i]))
        m.update(z)
    return np.array(err)


kw = dict(n_pos=2, scale_by_size=False)
models = {
    "cv q_vel=0.3": lambda z: KalmanCV(z, q_vel=0.3, **kw),
    "cv q_vel=0.1": lambda z: KalmanCV(z, q_vel=0.1, **kw),
    "ca": lambda z: KalmanCA(z, q_vel=0.3, q_acc=0.5, **kw),
    "imm default": lambda z: IMM(z, q_vel=0.3, q_acc=0.5, **kw),
    "imm cv+ca": lambda z: IMM(z, modes=(("cv", 0), ("ca", 0)), q_vel=0.3, q_acc=0.5, **kw),
    "imm cv+ct": lambda z: IMM(z, modes=(("cv", 0), ("ct", 0.12), ("ct", -0.12)), q_vel=0.3, **kw),
    "particle": lambda z: ParticleFilter(z, n_particles=500, sigma_acc=0.6, sigma_meas=0.3),
}
trajs = {"straight": straight(), "sharp_cut": sharp_cut(), "curve": curve(), "stop": stop()}
print(f"{'model':16s}" + "".join(f"{k:>12s}" for k in trajs) + "   (mean 1-step prediction error, px)")
for name, fn in models.items():
    row = []
    for tname, truth in trajs.items():
        e = np.mean([run(fn, truth, seed=s).mean() for s in range(5)])
        row.append(e)
    print(f"{name:16s}" + "".join(f"{v:12.3f}" for v in row))
