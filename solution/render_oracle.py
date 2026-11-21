"""Render the reference model flying both scored flight missions.

The video replays the scored flight test procedure from the public protocol:
for Test 1 and then Test 2 it shows the settle phase, the arm deployment
stage(s) under the arm PID, take-off and hover under the grader-supplied
cascaded flight controller (with the commanded heading step halfway through
the hover), the landing ramp, and the post-landing settle with the motors cut.
The camera tracks the drone so that the arm pose and attitude stay readable
during the 3 m climb.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from render_config import (
    DECLARED_YAW_COEFF,
    CascadedFlightController,
    body_tilt_deg,
    body_yaw,
    load_spec,
)

FPS = 25
WIDTH, HEIGHT = 1280, 720


def mission_profiles(ft):
    r = math.radians
    t1, t2 = ft["test_1"], ft["test_2"]
    return [
        {
            "name": "Test 1",
            "final": [r(t1["joint0_target"]), r(t1["joint1_target"]), r(t1["joint2_target"])],
            "stages": [
                (ft["arm_target_duration_1"],
                 [r(t1["joint0_target"]), r(t1["joint1_target"]), r(t1["joint2_target"])]),
            ],
        },
        {
            "name": "Test 2",
            "final": [r(t2["joint0_target_2"]), r(t2["joint1_target_2"]), r(t2["joint2_target_2"])],
            "stages": [
                (ft["arm_target_duration_1"], [None, r(t2["joint1_target_1"]), None]),
                (ft["arm_target_duration_2"],
                 [r(t2["joint0_target_2"]), r(t2["joint1_target_2"]), r(t2["joint2_target_2"])]),
            ],
        },
    ]


class Recorder:
    """Steps physics and emits one frame every 1/FPS s of simulated time."""

    def __init__(self, model, data, renderer, encoder, drone_id):
        self.model, self.data = model, data
        self.renderer, self.encoder = renderer, encoder
        self.drone_id = drone_id
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        # Chase shot: the camera follows the drone so the arm pose and attitude
        # stay readable; altitude is shown by a thin vertical gauge drawn from
        # the ground to the drone (a render-only decoration, not a model geom).
        self.camera.distance = 3.0
        self.camera.azimuth = 135
        self.camera.elevation = -12
        self.lookat = None
        self.next_frame_t = 0.0
        self.video_t = 0.0
        self.log = []

    def reset_camera(self):
        self.lookat = None

    def step(self, phase, mission, extra=None):
        mujoco.mj_step(self.model, self.data)
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError("Non-finite state during the oracle rollout")
        if self.data.time + 1e-9 >= self.next_frame_t:
            self.next_frame_t += 1.0 / FPS
            self._frame(phase, mission, extra)

    def _altitude_gauge(self, p):
        scn = self.renderer.scene
        if scn.ngeom + 12 > scn.maxgeom:
            return
        hub_bottom = float(p[2]) - 0.02
        # vertical gauge from the ground to the hub
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.array([1.0, 0.55, 0.0, 0.9], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006,
                             np.array([p[0], p[1], 0.0]), np.array([p[0], p[1], hub_bottom]))
        scn.ngeom += 1
        # 0.5 m tick marks along the gauge
        for k in range(1, 9):
            z = 0.5 * k
            if z > hub_bottom:
                break
            t = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(t, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32))
            mujoco.mjv_connector(t, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008,
                                 np.array([p[0] - 0.06, p[1], z]), np.array([p[0] + 0.06, p[1], z]))
            scn.ngeom += 1

    def _frame(self, phase, mission, extra):
        p = self.data.xpos[self.drone_id]
        target = np.array([p[0], p[1], p[2] - 0.2])
        # light smoothing keeps the camera steady while following the climb
        self.lookat = target if self.lookat is None else 0.85 * self.lookat + 0.15 * target
        self.camera.lookat[:] = self.lookat
        self.renderer.update_scene(self.data, camera=self.camera)
        # OSMesa shadow maps produce speckle on the ground plane; keep them off.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self._altitude_gauge(p)
        self.encoder.stdin.write(self.renderer.render().tobytes())
        frame_idx = len(self.log)
        if frame_idx % FPS == 0:
            q = np.degrees(self.data.qpos[7:10]) if self.model.nq >= 10 else []
            rec = {
                "video_t": round(frame_idx / FPS, 2),
                "mission": mission,
                "phase": phase,
                "sim_t": round(float(self.data.time), 3),
                "xyz": [round(float(v), 4) for v in p],
                "tilt_deg": round(body_tilt_deg(self.data, self.drone_id), 3),
                "yaw_deg": round(math.degrees(body_yaw(self.data, self.drone_id)), 3),
                "arm_deg": [round(float(v), 2) for v in q],
            }
            if extra:
                rec.update(extra)
            print(json.dumps(rec), file=sys.stderr, flush=True)
            self.log.append(rec)
        else:
            self.log.append(None)


def run_mission(rec, model, data, spec, profile, joint_q, joint_v, arm_act, rotors):
    ft = spec["flight_test"]
    cfg = ft["controller"]
    dt = model.opt.timestep
    kp, ki, kd = ft["pid_arm"]["Kp"], ft["pid_arm"]["Ki"], ft["pid_arm"]["Kd"]
    name = profile["name"]
    mujoco.mj_resetData(model, data)
    rec.next_frame_t = 0.0
    rec.reset_camera()
    integ = [0.0, 0.0, 0.0]
    did = rec.drone_id

    for _ in range(int(ft["settle_time"] / dt)):
        for a in arm_act:
            data.ctrl[a] = 0.0
        rec.step("settle", name)
    p0 = data.xpos[did].copy()
    z0 = p0[2]

    def arm(targets):
        for j in range(3):
            if targets[j] is None:
                continue
            e = targets[j] - data.qpos[joint_q[j]]
            integ[j] += e * dt
            u = kp * e + ki * integ[j] - kd * data.qvel[joint_v[j]]
            data.ctrl[arm_act[j]] = max(-1.0, min(1.0, u))

    stage_ok = True
    for k, (duration, targets) in enumerate(profile["stages"]):
        for _ in range(int(duration / dt)):
            arm(targets)
            rec.step(f"arm stage {k + 1}", name)
        for j in range(3):
            if targets[j] is not None and abs(targets[j] - data.qpos[joint_q[j]]) > math.radians(5.0):
                stage_ok = False

    ctl = CascadedFlightController(model, did, rotors, cfg)
    if not ctl.ok:
        raise RuntimeError("flight controller could not be built for the oracle model")
    n_hover = int(ft["hover_time"] / dt)
    z_t = z0 + ft["target_z_displacement"]
    yaw0 = body_yaw(data, did)
    yaw1 = yaw0 + math.radians(cfg["yaw_step_deg"])
    max_z, max_tilt = -1e9, 0.0
    for s in range(n_hover):
        arm(profile["final"])
        yaw_cmd = yaw0 if s < n_hover // 2 else yaw1
        ctl.command(data, z_t, p0[:2], yaw_cmd, log=s >= n_hover // 2)
        rec.step("take-off/hover" if s < n_hover // 2 else "hover + heading step", name)
        max_z = max(max_z, data.xpos[did][2])
        max_tilt = max(max_tilt, body_tilt_deg(data, did))
    yaw_err = (body_yaw(data, did) - yaw1 + math.pi) % (2 * math.pi) - math.pi

    n_land = int(ft["landing_time"] / dt)
    for s in range(n_land):
        arm(profile["final"])
        zc = z_t + (z0 - z_t) * (s + 1) / n_land
        ctl.command(data, zc, p0[:2], yaw1, log=False)
        rec.step("landing", name)
        max_tilt = max(max_tilt, body_tilt_deg(data, did))
    for r in rotors:
        data.ctrl[r] = 0.0
    for _ in range(int(1.0 / dt)):
        rec.step("motors cut", name)
    pf = data.xpos[did]
    summary = {
        "mission": name,
        "stage_joints_converged": stage_ok,
        "max_hover_dz": round(float(max_z - z0), 4),
        "yaw_err_deg_end_hover": round(math.degrees(yaw_err), 3),
        "rotor_saturated_steps": ctl.saturated_steps,
        "max_tilt_deg": round(max_tilt, 3),
        "landed_dz": round(float(pf[2] - z0), 4),
        "xy_drift": round(float(np.linalg.norm(pf[:2] - p0[:2])), 4),
    }
    print(json.dumps({"summary": summary}), file=sys.stderr, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/output"))
    parser.add_argument("--metrics", type=Path, default=None,
                        help="optional path for the 1 Hz rollout metrics (JSON)")
    args = parser.parse_args()
    spec = load_spec()
    model = mujoco.MjModel.from_xml_path(str(args.output_dir / "model.xml"))
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT

    did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    jn = ["arm_joint0", "arm_joint1", "arm_joint2"]
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
    joint_q = [int(model.jnt_qposadr[j]) for j in jid]
    joint_v = [int(model.jnt_dofadr[j]) for j in jid]
    arm_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in ["joint0_motor", "joint1_motor", "joint2_motor"]]
    sites = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in DECLARED_YAW_COEFF}
    rotors = [i for i in range(model.nu)
              if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_SITE
              and model.actuator_trnid[i, 0] in sites]
    if len(rotors) != 4 or did < 0 or min(jid) < 0 or min(arm_act) < 0:
        raise RuntimeError("oracle model is missing the scored drone/arm/rotor elements")

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output_dir / "rendering.mp4"),
    ]
    summaries = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            rec = Recorder(model, data, renderer, encoder, did)
            for profile in mission_profiles(spec["flight_test"]):
                summaries.append(run_mission(rec, model, data, spec, profile,
                                             joint_q, joint_v, arm_act, rotors))
        finally:
            encoder.stdin.close()
            returncode = encoder.wait()
        if returncode:
            raise RuntimeError(f"ffmpeg exited with status {returncode}")
    if args.metrics is not None:
        args.metrics.write_text(json.dumps(
            {"fps": FPS, "summaries": summaries,
             "per_second": [r for r in rec.log if r is not None]}, indent=1))


if __name__ == "__main__":
    main()
