"""Shared helpers for the reviewer render of the scored flight missions.

The flight controller below is a verbatim copy of the grader-supplied cascaded
controller that the public protocol documents (instruction.md / evaluation
protocol "flight_test"); the render keeps its own copy so it never imports the
private scorer tree.  Gains and limits are read from data/spec.json, so the
video must be regenerated whenever spec.json changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DECLARED_YAW_COEFF = {
    "thruster_front": 0.1,
    "thruster_back": 0.1,
    "thruster_left": -0.1,
    "thruster_right": -0.1,
}


def mujoco():
    import mujoco
    return mujoco


def load_spec():
    candidates = [
        Path("/data/spec.json"),
        Path(__file__).resolve().parent.parent / "data" / "spec.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError("spec.json not found in /data or <task>/data")


def initialize(model, data):
    mj = mujoco()
    mj.mj_resetData(model, data)
    for i in range(model.njnt):
        if int(model.jnt_type[i]) == mj.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[i]] = 0
    mj.mj_forward(model, data)


def body_yaw(data, body_id):
    m = data.xmat[body_id]
    return math.atan2(m[3], m[0])


def body_tilt_deg(data, body_id):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body_id][8])))))


class CascadedFlightController:
    """XY PD -> roll/pitch setpoints; altitude PD + tilt-compensated feed-forward;
    attitude/yaw PD on body rates; 4x4 mixer from the thruster site positions."""

    def __init__(self, model, drone_body_id, rotor_actuator_ids, cfg):
        mj = mujoco()
        self.cfg = cfg
        self.did = drone_body_id
        self.rot = list(rotor_actuator_ids)
        self.mass = mj.mj_getTotalmass(model)
        self.dof = int(model.jnt_dofadr[model.body_jntadr[drone_body_id]])
        self.saturated_steps = 0
        self.ok = True
        A = np.zeros((4, 4))
        for k, a in enumerate(self.rot):
            s_id = model.actuator_trnid[a, 0]
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, s_id)
            if name not in DECLARED_YAW_COEFF or model.site_bodyid[s_id] != drone_body_id:
                self.ok = False
                return
            p = model.site_pos[s_id]
            A[:, k] = [1.0, p[1], -p[0], DECLARED_YAW_COEFF[name]]
        if abs(np.linalg.det(A)) < 1e-9:
            self.ok = False
            return
        self.Ainv = np.linalg.inv(A)

    def command(self, data, z_target, xy_target, yaw_target, log=False):
        c = self.cfg
        g = 9.81
        R = data.xmat[self.did].reshape(3, 3)
        p = data.xpos[self.did]
        v = data.qvel[self.dof:self.dof + 3]
        w = data.qvel[self.dof + 3:self.dof + 6]
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        yaw = math.atan2(R[1, 0], R[0, 0])
        ax = c["xy_pd"]["Kp"] * (xy_target[0] - p[0]) - c["xy_pd"]["Kd"] * v[0]
        ay = c["xy_pd"]["Kp"] * (xy_target[1] - p[1]) - c["xy_pd"]["Kd"] * v[1]
        cy, sy = math.cos(yaw), math.sin(yaw)
        axb = cy * ax + sy * ay
        ayb = -sy * ax + cy * ay
        lim = math.radians(c["tilt_cmd_max_deg"])
        pitch_d = max(-lim, min(lim, axb / g))
        roll_d = max(-lim, min(lim, -ayb / g))
        az = c["pd_z"]["Kp"] * (z_target - p[2]) - c["pd_z"]["Kd"] * v[2]
        collective = self.mass * (g + az) / max(0.5, R[2, 2])
        ka, kd = c["attitude_pd"]["Kp"], c["attitude_pd"]["Kd"]
        tx = ka * (roll_d - roll) - kd * w[0]
        ty = ka * (pitch_d - pitch) - kd * w[1]
        ey = (yaw_target - yaw + math.pi) % (2 * math.pi) - math.pi
        tz = c["yaw_pd"]["Kp"] * ey - c["yaw_pd"]["Kd"] * w[2]
        f = self.Ainv @ np.array([collective, tx, ty, tz])
        lo, hi = c["thrust_range"]
        fc = np.clip(f, lo, hi)
        if log and np.any(np.abs(f - fc) > 1e-12):
            self.saturated_steps += 1
        for a, u in zip(self.rot, fc):
            data.ctrl[a] = float(u)
