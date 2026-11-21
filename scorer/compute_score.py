"""Deterministic MuJoCo grader for an Aerial Manipulator.

Authored as a declarative RubricTask to evaluate physical and structural
constraints of the generated MJCF model (Quadrotor + 3-DOF Arm).
"""

from __future__ import annotations

import json
from logging import root
import math
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from grading.evaluation import (
    RubricCriterion,
    RubricTask,
    TextArtifact,
)

ROLLOUT_DURATION_SEC = 2.0

def _mujoco():
    import mujoco
    return mujoco


def _load_spec():
    production_path = Path("/data/spec.json")
    local_path = Path(__file__).parent.parent / "data" / "spec.json"
    for path in (production_path, local_path):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if  "mass" in data and "joints_range" in data:
            return data
    raise FileNotFoundError("No valid spec.json found (checked /data and local fallback)")


def _load_model(source):
    """Attempts to parse XML source text into a compiled MuJoCo MjModel."""
    mj = _mujoco()
    return mj.MjModel.from_xml_string(source)



# Pre-compile structural gate (v1 grading limits). MuJoCo's XML compile and
# step are native calls that no Python-level wall can interrupt, so a model
# whose compile or simulation cost is unbounded (a large <flex>, an <hfield>
# or <mesh> asset, a plugin) could stall the grade past its platform wait and
# be discarded instead of scored. The task needs none of these features, so
# the raw XML is checked before compilation and such a model scores 0.
_PRECOMPILE_BANNED_TAGS = frozenset({
    "include", "replicate", "composite", "flexcomp", "flex", "deformable",
    "skin", "plugin", "extension", "instance", "hfield", "mesh", "texture",
    "size", "frame", "attach", "model",
})
_PRECOMPILE_MAX_ELEMENTS = 2000
_PRECOMPILE_MAX_ATTR_CHARS = 4096


def _precompile_gate(xml_string):
    """True when the raw MJCF stays inside the task's bounded feature set."""
    if not isinstance(xml_string, str):
        return False
    lowered = xml_string.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        return False
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return False
    if root.tag != "mujoco":
        return False
    count = 0
    for element in root.iter():
        count += 1
        if count > _PRECOMPILE_MAX_ELEMENTS:
            return False
        if not isinstance(element.tag, str):
            continue
        if element.tag.lower() in _PRECOMPILE_BANNED_TAGS:
            return False
        for name, value in element.attrib.items():
            if name.lower() in {"file", "content_type"}:
                return False
            if len(value) > _PRECOMPILE_MAX_ATTR_CHARS:
                return False
    return True


def _postcompile_gate(model):
    """The compiled model carries no unbounded-cost feature the gate missed."""
    return (
        int(model.nflex) == 0
        and int(model.nplugin) == 0
        and int(model.nskin) == 0
        and int(model.nhfield) == 0
        and int(model.nmesh) == 0
        and int(model.ntex) == 0
    )


def _all_failed():
    return {criterion: False for criterion in _CRITERION_NAMES}


def _check_base_and_frame_structure(model, spec):
    """
    Verifies the drone's Base & Frame requirements:
    - Body 1 is named 'drone' at pos [0, 0, 1.0] with a free joint.
    - 'central_hub' geom is a cylinder at local pos [0, 0, 0].
    - 'central_hub' and all four arms have a height (Z-dimension) complying with
      spec['central_hub_height'].
    - Four named arms extending in exactly +X, -X, +Y, -Y directions, with identical
      geometry type, identical sizes across all axes, and at Z = 0 (same as central hub).
    - All four arms must be symmetric in distance from the hub.
    - When the drone is dropped from 1m, at most 2 geoms may contact the ground.
    """
    mj = _mujoco()
    
    # 1. Check Body ID 1 name and position
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1)
    if body_name != "drone":
        return False
        
    if not np.allclose(model.body_pos[1], [0.0, 0.0, 1.0], atol=1e-4):
        return False
        
    # 2. Check for a Free Joint in Body 1
    has_free_joint = False
    jnt_start = model.body_jntadr[1]
    jnt_num = model.body_jntnum[1]
    for j_idx in range(jnt_start, jnt_start + jnt_num):
        if model.jnt_type[j_idx] == mj.mjtJoint.mjJNT_FREE:
            has_free_joint = True
            break
    if not has_free_joint:
        return False
        
    # 3. Check Central Hub Geometry and Height
    hub_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "central_hub")
    if hub_id == -1:
        return False
    if model.geom_bodyid[hub_id] != 1:  # Must belong to the drone body
        return False
    if model.geom_type[hub_id] != mj.mjtGeom.mjGEOM_CYLINDER:
        return False
    if not np.allclose(model.geom_pos[hub_id], [0.0, 0.0, 0.0], atol=1e-4):
        return False

    # Height check for central_hub (for vertical cylinder, geom_size[1] is half-height)
    hub_height = 2.0 * model.geom_size[hub_id][1]
    target_height = spec["central_hub_height"]["target"]
    tol_height = spec["central_hub_height"]["tolerance"]
    if abs(hub_height - target_height) > tol_height:
        return False

    # 4. Check the Four Arms
    arm_names = ["arm_front", "arm_back", "arm_left", "arm_right"]
    arm_ids = []
    for name in arm_names:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            return False
        if model.geom_bodyid[gid] != 1:
            return False
        arm_ids.append(gid)

    # 5. Check Arms Geometry Type, All Dimensions, Z-Level (Z=0), and Height Constraint
    geom_type = model.geom_type[arm_ids[0]]
    geom_size = model.geom_size[arm_ids[0]]
    
    # Compute arm height (Z-dimension) based on geometry type
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        return False
    else:
        # For horizontal capsules, radius (geom_size[0]) determines Z-height
        arm_height = 2.0 * geom_size[0]

    # Arm height must also satisfy the central_hub_height tolerance window
    if abs(arm_height - target_height) > tol_height:
        return False

    for gid in arm_ids:
        # Z-level must be identical to central hub level (Z = 0)
        if not math.isclose(model.geom_pos[gid][2], 0.0, abs_tol=1e-4):
            return False
        # Must be the exact same geometry type (e.g. all capsules)
        if model.geom_type[gid] != geom_type:
            return False
        # Must have identical size dimensions across all axes
        if not np.allclose(model.geom_size[gid], geom_size, atol=1e-4):
            return False

    # 6. Check Directions (Centers of the geoms must sit directly on the axes)
    pos_f = model.geom_pos[arm_ids[0]]
    pos_b = model.geom_pos[arm_ids[1]]
    pos_l = model.geom_pos[arm_ids[2]]
    pos_r = model.geom_pos[arm_ids[3]]

    # Ensure strictly on the X or Y axis with the correct direction
    if pos_f[0] <= 0 or abs(pos_f[1]) > 1e-4: return False  # Front: +X
    if pos_b[0] >= 0 or abs(pos_b[1]) > 1e-4: return False  # Back: -X
    if pos_l[1] <= 0 or abs(pos_l[0]) > 1e-4: return False  # Left: +Y
    if pos_r[1] >= 0 or abs(pos_r[0]) > 1e-4: return False  # Right: -Y

    # Ensure symmetric distance from the hub
    dist_f = pos_f[0]
    dist_b = -pos_b[0]
    dist_l = pos_l[1]
    dist_r = -pos_r[1]

    if not (math.isclose(dist_f, dist_b, abs_tol=1e-4) and 
            math.isclose(dist_f, dist_l, abs_tol=1e-4) and 
            math.isclose(dist_f, dist_r, abs_tol=1e-4)):
        return False

    # 7. Check Ground Contact Limit (Max 2 geoms in contact with ground when settled)
    # Find the ground plane geom ID
    ground_geom_id = -1
    for i in range(model.ngeom):
        if model.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE:
            ground_geom_id = i
            break
            
    if ground_geom_id != -1:
        data = mj.MjData(model)
        # Step the simulation to let the drone fall and settle (2000 steps = 2 seconds at default 0.001 timestep)
        settle_time = 2.0  # seconds
        steps = int(settle_time / model.opt.timestep)
        for _ in range(steps):
            mj.mj_step(model, data)
            
        # Count unique geoms touching the ground from the entire system
        geoms_touching_ground = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == ground_geom_id:
                geoms_touching_ground.add(g2)
            elif g2 == ground_geom_id:
                geoms_touching_ground.add(g1)
                
        if len(geoms_touching_ground) > 2:
            return False

    return True



def _segment_distance(p1, p2, p3, p4):
    """Calculates the exact shortest distance between two 3D line segments."""
    EPS = 1e-8
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a <= EPS and e <= EPS:
        return np.linalg.norm(r)
        
    if a <= EPS:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
    else:
        c = np.dot(d1, r)
        if e <= EPS:
            t = 0.0
            s = -c / a
            s = max(0.0, min(1.0, s))
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            
            if denom > EPS:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
                
            t = (b * s + f) / e
            
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))

    c1 = p1 + d1 * s
    c2 = p3 + d2 * t
    return np.linalg.norm(c1 - c2)

def _test_segment_distance_symmetry():
    """Internal regression test ensuring XML order doesn't affect distance calculations."""
    p1, p2 = np.array([0, 0, 0]), np.array([2, 0, 0])
    p3, p4 = np.array([3, 1, 0]), np.array([5, 1, 0])
    
    dist_ab = _segment_distance(p1, p2, p3, p4)
    dist_ba = _segment_distance(p3, p4, p1, p2)
    
    if not np.isclose(dist_ab, dist_ba):
        raise RuntimeError(f"Contiguity math asymmetry detected: {dist_ab} != {dist_ba}")

def _get_geom_endpoints(model, mj, geom_id):
    """Returns the two endpoints and radius of a capsule/cylinder."""
    pos = model.geom_pos[geom_id]
    quat = model.geom_quat[geom_id]
    mat = np.zeros(9)
    mj.mju_quat2Mat(mat, quat)
    z_axis = mat.reshape(3, 3)[:, 2]
    
    half_len = model.geom_size[geom_id][1]
    radius = model.geom_size[geom_id][0]
    
    p1 = pos + (z_axis * half_len)
    p2 = pos - (z_axis * half_len)
    return p1, p2, radius

def _check_drone_contiguity(model):
    """Verifies physical contiguity using exact capsule line-segment distances.

    The structure is every geom rigidly welded to the ``drone`` body (geoms of
    ``drone`` and of any jointless child body) plus the geoms of any extra
    jointed child body of ``drone`` other than the manipulator links. Positions
    are evaluated in the world frame at the default pose (qpos0). Every one of
    these geoms must be reachable from ``central_hub`` through a chain of
    touching geoms (segment distance <= r1 + r2 + 1 mm).
    """
    _test_segment_distance_symmetry()

    mj = _mujoco()
    drone_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
    if drone_id == -1:
        return False
    link_ids = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link1"),
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2"),
    }
    drone_weld = model.body_weldid[drone_id]

    def in_structure(body):
        if body <= 0 or model.body_rootid[body] != drone_id:
            return False
        if model.body_weldid[body] == drone_weld:
            return True
        # extra (jointed) body: any drone-subtree body that is not a manipulator link
        # and is not below a manipulator link
        b = body
        while b != drone_id and b > 0:
            if b in link_ids:
                return False
            b = model.body_parentid[b]
        return b == drone_id

    gids = [g for g in range(model.ngeom) if in_structure(int(model.geom_bodyid[g]))]
    hub_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "central_hub")
    if hub_id == -1 or hub_id not in gids:
        return False

    valid_types = (mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER)
    if any(model.geom_type[g] not in valid_types for g in gids):
        return False

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_kinematics(model, data)

    def endpoints(g):
        pos = data.geom_xpos[g]
        z_axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
        half_len = model.geom_size[g][1]
        return pos + z_axis * half_len, pos - z_axis * half_len, model.geom_size[g][0]

    ends = {g: endpoints(g) for g in gids}
    adj_list = {g: [] for g in gids}
    margin = 1e-3
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            g1, g2 = gids[i], gids[j]
            p1, p2, r1 = ends[g1]
            p3, p4, r2 = ends[g2]
            if _segment_distance(p1, p2, p3, p4) <= (r1 + r2 + margin):
                adj_list[g1].append(g2)
                adj_list[g2].append(g1)

    visited = set()
    queue = [hub_id]
    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            queue.extend(n for n in adj_list[current] if n not in visited)
    return len(visited) == len(gids)




# Thruster site local z-axis must be vertical: cos(tilt) >= 1 - 1e-6 (tilt <= ~0.08 deg).
_THRUSTER_Z_MIN = 1.0 - 1e-6


def _check_rotors_and_propellers(model, spec):
    """
    Verifies rotors and propellers constraints:
    - 4 named thruster sites mounted correctly on the arms.
    - Propeller clearance ensures 36 cm blades won't overlap.
    - Exactly 4 flight motors with strictly defined gear vectors.
    - Actuators must be pure motors (gaintype FIXED, biastype NONE).
    - Thruster sites must point their local Z-axis up.
    """
    mj = _mujoco()
    
    site_specs = {
        "thruster_front": {"arm": "arm_front", "axis": 0, "sign": 1, "gear": [0, 0, 1, 0, 0, 0.1]},
        "thruster_back": {"arm": "arm_back", "axis": 0, "sign": -1, "gear": [0, 0, 1, 0, 0, 0.1]},
        "thruster_left": {"arm": "arm_left", "axis": 1, "sign": 1, "gear": [0, 0, 1, 0, 0, -0.1]},
        "thruster_right": {"arm": "arm_right", "axis": 1, "sign": -1, "gear": [0, 0, 1, 0, 0, -0.1]},
    }
    
    site_positions = []
    
    for site_name, site_cfg in site_specs.items():
        # 1. Verify Site Exists
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        if sid == -1: 
            return False

        # Ensure the site is actually parented to the drone body
        drone_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
        if model.site_bodyid[sid] != drone_body_id:
            return False
            
        s_pos = model.site_pos[sid]
        site_positions.append(s_pos)
        
        # 2. Verify Arm Alignment and Height
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, site_cfg["arm"])
        if aid == -1: 
            return False
        a_pos = model.geom_pos[aid]
        
        # Calculate world-frame extent (half-length) along the target axis
        g_type = model.geom_type[aid]
        size = model.geom_size[aid]
        
        # Get the rotation matrix of the geometry
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, model.geom_quat[aid])
        mat = mat.reshape(3, 3)
        
        k = site_cfg["axis"] # 0 for X (front/back), 1 for Y (left/right)
        
        # Project local sizes onto the global measurement axis
        if g_type == mj.mjtGeom.mjGEOM_BOX:
            half_length = size[0] * abs(mat[k, 0]) + size[1] * abs(mat[k, 1]) + size[2] * abs(mat[k, 2])
        elif g_type == mj.mjtGeom.mjGEOM_CAPSULE:
            half_length = size[0] + size[1] * abs(mat[k, 2])
        elif g_type == mj.mjtGeom.mjGEOM_CYLINDER:
            half_length = size[1] * abs(mat[k, 2]) + size[0] * np.linalg.norm([mat[k, 0], mat[k, 1]])
        else:
            half_length = model.geom_rbound[aid]

        arm_length = abs(a_pos[k]) + half_length
        other_axis = 1 if k == 0 else 0
        
        if abs(s_pos[other_axis]) > 1e-4: 
            return False
        
        if np.sign(s_pos[k]) != site_cfg["sign"] and abs(s_pos[k]) > 1e-4: 
            return False
            
        if abs(s_pos[k]) > arm_length + 1e-4: 
            return False
            
        z_diff = s_pos[2] - a_pos[2]
        if z_diff < -1e-4 or z_diff > spec["rotor_sites_height_diff_max"] + 1e-4: 
            return False
            
        # 3. Verify Actuators & Gear Vectors
        motor_found = False
        for i in range(model.nu):
            if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_SITE:
                if model.actuator_trnid[i, 0] == sid:
                    motor_found = True
                    gear = model.actuator_gear[i]
                    if not np.allclose(gear, site_cfg["gear"], atol=1e-4):
                        return False
                        
                    # Enforce pure motor dynamics (No <position> or <general> actuator cheats)
                    if model.actuator_gaintype[i] != mj.mjtGain.mjGAIN_FIXED:
                        return False
                    if abs(model.actuator_gainprm[i, 0] - 1.0) > 1e-4:
                        return False
                        
                    if model.actuator_biastype[i] != mj.mjtBias.mjBIAS_NONE:
                        return False
                    if np.any(np.abs(model.actuator_biasprm[i]) > 1e-4):
                        return False
                    
                    break
                    
        if not motor_found:
            return False

        # NEW: Convert the site's local quaternion to a 3x3 matrix to check orientation
        site_mat = np.zeros(9)
        mj.mju_quat2Mat(site_mat, model.site_quat[sid])
        
        # In a flattened 3x3 matrix (row-major), index 8 is the Z-component of the local Z-axis.
        z_up_component = site_mat[8]
        if z_up_component < _THRUSTER_Z_MIN:
            return False
        
    # 4. Anti-cheat: Ensure there are exactly 4 site-driven actuators
    flight_actuators = sum(1 for i in range(model.nu) if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_SITE)
    if flight_actuators != 4:
        return False
        
    # 5. Propeller Clearance Check
    for i in range(len(site_positions)):
        for j in range(i + 1, len(site_positions)):
            dist = np.linalg.norm(site_positions[i] - site_positions[j])
            if dist < spec["propellers_diameter"]:
                return False
                
    return True




def _check_manipulator_topology(model, spec):
    """
    Verifies Manipulator Topology constraints:
    - Bodies 'link1' (child of 'drone') and 'link2' (child of 'link1') connected in series.
    - 'link2' has no downstream child bodies.
    - Both 'link1' and 'link2' meet the minimum mass requirement.
    - Attachment of 'link1' is vertically below central hub within spec['manipulator']['base_pos_to_hub_range'].
    - link1 has exactly 1 geom (capsule). 
    - link2's first geom is a capsule, is the longest in link2, and acts as its primary geom.
    - Primary capsule geometries length >= spec['manipulator']['link_length_min'].
    - Primary capsule length of link2 - length of link1 >= spec['manipulator']['links_length_diff_min'].
    - Manipulator is structurally mounted via 'arm_base' capsule to a hub-connected capsule.
    """
    mj = _mujoco()
    
    # 1. Check Body Existence and Hierarchy
    drone_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
    link1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link1")
    link2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
    
    if drone_id == -1 or link1_id == -1 or link2_id == -1: return False
    if model.body_parentid[link1_id] != drone_id: return False
    if model.body_parentid[link2_id] != link1_id: return False
        
    for b_idx in range(model.nbody):
        if model.body_parentid[b_idx] == link2_id: return False
        if model.body_parentid[b_idx] == link1_id and b_idx != link2_id: return False

    # 2. Check Link Masses
    min_mass = spec["manipulator"]["links_mass_min"]
    if model.body_mass[link1_id] < min_mass : return False
    if model.body_mass[link2_id] < min_mass : return False


    # 3. Check Base Position (link1 position relative to drone central hub)
    link1_pos = model.body_pos[link1_id]
    if link1_pos[2] >= 0: return False
        
    vert_dist = -link1_pos[2]
    min_dist, max_dist = spec["manipulator"]["base_pos_to_hub_range"]
    if not (min_dist - 1e-4 <= vert_dist <= max_dist + 1e-4): return False

    # 4. Check Geometries and Link Lengths
    gids_link1 = [g for g in range(model.ngeom) if model.geom_bodyid[g] == link1_id]
    gids_link2 = [g for g in range(model.ngeom) if model.geom_bodyid[g] == link2_id]
    
    # Strictly enforce link1 has exactly one geometry, and it is a capsule
    if len(gids_link1) != 1: return False
    primary_l1 = gids_link1[0]
    if model.geom_type[primary_l1] != mj.mjtGeom.mjGEOM_CAPSULE: return False
    
    # Enforce link2 has at least one geom, and the first one is a capsule (primary)
    if not gids_link2: return False
    primary_l2 = gids_link2[0]
    if model.geom_type[primary_l2] != mj.mjtGeom.mjGEOM_CAPSULE: return False
    
    length_l1 = 2.0 * model.geom_size[primary_l1][1]
    length_l2 = 2.0 * model.geom_size[primary_l2][1]
    
    # Check that primary_l2 is the longest geometry in link2
    for g in gids_link2[1:]:
        # Using size[1] to approximate length for elongated types, or max size for others
        if model.geom_type[g] in [mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER]:
            g_len = 2.0 * model.geom_size[g][1]
        else:
            g_len = 2.0 * max(model.geom_size[g])
            
        if g_len > length_l2: 
            return False

    min_length = spec["manipulator"]["link_length_min"]
    
    # Check minimum length requirement for both links
    if length_l1 < min_length - 1e-4 or length_l2 < min_length - 1e-4: return False

    # Check length difference requirement: length_l2 - length_l1 >= links_length_diff_min
    links_diff_min = spec["manipulator"]["links_length_diff_min"]
    if (length_l2 - length_l1) < links_diff_min - 1e-4: return False

    # 5. Check that link2's origin is strictly at the distal tip of link1's capsule
    c1 = model.geom_pos[primary_l1]
    
    quat1 = model.geom_quat[primary_l1]
    mat1 = np.zeros(9)
    mj.mju_quat2Mat(mat1, quat1)
    z_axis1 = mat1.reshape(3, 3)[:, 2]
    
    half_len1 = model.geom_size[primary_l1][1]
    
    end1_a = c1 + (z_axis1 * half_len1)
    end1_b = c1 - (z_axis1 * half_len1)
    
    # Identify the distal tip (the endpoint farthest from link1's base origin [0, 0, 0])
    distal_tip1 = end1_a if np.linalg.norm(end1_a) > np.linalg.norm(end1_b) else end1_b
    
    link2_pos = model.body_pos[link2_id]
    
    # Reject if link2 is placed anywhere other than the far tip of link1
    if not np.allclose(link2_pos, distal_tip1, atol=1e-4):
        return False

    # Verify primary geometry of link2 starts from "0 0 0" in link2's local frame
    c2 = model.geom_pos[primary_l2]
    
    quat2 = model.geom_quat[primary_l2]
    mat2 = np.zeros(9)
    mj.mju_quat2Mat(mat2, quat2)
    z_axis2 = mat2.reshape(3, 3)[:, 2]
    
    half_len2 = model.geom_size[primary_l2][1]
    
    end2_a = c2 + (z_axis2 * half_len2)
    end2_b = c2 - (z_axis2 * half_len2)
    
    if not (np.allclose(end2_a, [0, 0, 0], atol=1e-4) or np.allclose(end2_b, [0, 0, 0], atol=1e-4)):
        return False
    
    # 6. Check Manipulator Mounting (link1 inside arm_base) ---
    arm_base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "arm_base")
    if arm_base_id == -1 or model.geom_bodyid[arm_base_id] != drone_id:
        return False
    if model.geom_type[arm_base_id] != mj.mjtGeom.mjGEOM_CAPSULE:
        return False

    # Calculate endpoints and radius of arm_base
    c_base = model.geom_pos[arm_base_id]
    quat_base = model.geom_quat[arm_base_id]
    mat_base = np.zeros(9)
    mj.mju_quat2Mat(mat_base, quat_base)
    z_axis_base = mat_base.reshape(3, 3)[:, 2]
    
    half_len_base = model.geom_size[arm_base_id][1]
    radius_base = model.geom_size[arm_base_id][0]

    base_end1 = c_base + (z_axis_base * half_len_base)
    base_end2 = c_base - (z_axis_base * half_len_base)

    # Verify link1_pos lies inside or exactly on the surface of the arm_base capsule
    line_vec = base_end2 - base_end1
    line_len = np.linalg.norm(line_vec)
    
    if line_len > 1e-5:
        line_dir = line_vec / line_len
        vec_to_pt = link1_pos - base_end1
        
        # Project link1_pos onto the core line segment
        t = np.dot(vec_to_pt, line_dir)
        t_clamped = max(0.0, min(line_len, t))
        closest_pt = base_end1 + t_clamped * line_dir
        
        # Calculate exact distance from link1_pos to the line segment
        dist = np.linalg.norm(link1_pos - closest_pt)
        
        # If distance exceeds the radius, link1 is floating outside arm_base
        if dist > radius_base + 1e-4:
            return False
    else:
        # Fallback if arm_base has length ~0
        if np.linalg.norm(link1_pos - c_base) > radius_base + 1e-4:
            return False
                
    return True




def _check_manipulator_joints(model, spec):
    """
    Verifies Manipulator Joints & Actuators:
    - Exactly 3 hinge joints on link1 and link2 with pos="0 0 0".
    - Joint names: 'arm_joint0', 'arm_joint1', 'arm_joint2'.
    - Joint ownership: arm_joint0 and arm_joint1 on link1; arm_joint2 on link2.
    - Joint ranges match spec['joints_range'].
    - arm_joint0 is vertical (Z-axis), arm_joint1 is horizontal, arm_joint2 equals arm_joint1.
    - arm_joint1 and arm_joint2 must be strictly orthogonal to the longitudinal axis of their respective links.
    - Damping <= spec['damping_max'] and Frictionloss <= spec['frictionloss_max'].
    - Frictionloss strictly 0.
    - Exactly 3 motors named 'joint0_motor', 'joint1_motor', 'joint2_motor' bound to corresponding joints with gear=1.0 and ctrlrange=[-1, 1].
    - Actuators must be pure motors (gaintype FIXED, biastype NONE).
    """
    
    mj = _mujoco()

    # 1. Verify body existence
    link1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link1")
    link2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
    if link1_id == -1 or link2_id == -1: 
        return False

    # 2. Verify joint existence
    j0_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint0")
    j1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint1")
    j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint2")
    if j0_id == -1 or j1_id == -1 or j2_id == -1: 
        return False

    # 3. ENFORCE JOINT OWNERSHIP
    if model.jnt_bodyid[j0_id] != link1_id or model.jnt_bodyid[j1_id] != link1_id:
        return False
    if model.jnt_bodyid[j2_id] != link2_id:
        return False

    # 4. Gather all joints belonging to the manipulator
    manipulator_joints = []
    for bid in [link1_id, link2_id]:
        j_start = model.body_jntadr[bid]
        j_num = model.body_jntnum[bid]
        for j in range(j_start, j_start + j_num):
            manipulator_joints.append(j)
            
    if len(manipulator_joints) != 3: 
        return False
    if not set([j0_id, j1_id, j2_id]).issubset(set(manipulator_joints)): 
        return False

    # 5. Check joint types and local pos="0 0 0"
    for jid in [j0_id, j1_id, j2_id]:
        if model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE: 
            return False
        if not np.allclose(model.jnt_pos[jid], [0, 0, 0], atol=1e-4): 
            return False

    # 6. Check joint ranges
    def check_range(j_id, expected_degrees):
        rng = model.jnt_range[j_id]
        return (abs(math.degrees(rng[0]) - expected_degrees[0]) < 1.0 and 
                abs(math.degrees(rng[1]) - expected_degrees[1]) < 1.0)
                
    if not check_range(j0_id, spec["joints_range"]["shoulder_vertical"]): 
        return False
    if not check_range(j1_id, spec["joints_range"]["shoulder_horizontal"]): 
        return False
    if not check_range(j2_id, spec["joints_range"]["elbow"]): 
        return False

    # 7. Check Damping and Frictionloss
    max_damping = spec["joints_range"]["damping_max"]
    max_frictionloss = spec["joints_range"]["frictionloss_max"]
    for jid in [j0_id, j1_id, j2_id]:
        dof_idx = model.jnt_dofadr[jid]
        if model.dof_damping[dof_idx] > max_damping + 1e-4:
            return False
        if model.dof_frictionloss[dof_idx] > max_frictionloss + 1e-4:
            return False
    
    # 8. Check Joint Axes in World Coordinates
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)

    world_ax0 = data.xaxis[j0_id]
    world_ax1 = data.xaxis[j1_id]
    world_ax2 = data.xaxis[j2_id]

    # j0 (base joint) must be aligned with the world Z-axis
    if abs(world_ax0[2]) < 0.999:
        return False

    # j1 (shoulder joint) must lie in the horizontal plane (perpendicular to world Z-axis)
    if abs(world_ax1[2]) > 1e-4:
        return False

    # j1 and j2 must be parallel in world coordinates.
    # If parallel, their 3D cross product magnitude is 0.
    cross_prod = np.cross(world_ax1, world_ax2)
    if np.linalg.norm(cross_prod) > 1e-4:
        return False
    
    # Ensure they point in the exact same direction (dot product == 1)
    if np.dot(world_ax1, world_ax2) < 0.99:
        return False

    # --- Anti-roll loophole check ---
    # Both joint1 and joint2 must be orthogonal to their respective link's primary (longest) capsule.
    for link_id, j_id in [(link1_id, j1_id), (link2_id, j2_id)]:
        geom_start = model.body_geomadr[link_id]
        geom_num = model.body_geomnum[link_id]
        
        if geom_num == 0:
            return False

        # Identify the primary geom as the longest capsule on this link
        primary_geom_id = -1
        max_hlength = -1.0
        
        for i in range(geom_start, geom_start + geom_num):
            # For capsules in MuJoCo, geom_size[1] represents the half-length of the cylindrical segment
            hlength = model.geom_size[i, 1]
            if hlength > max_hlength:
                max_hlength = hlength
                primary_geom_id = i

        if primary_geom_id == -1:
            return False

        # Convert the primary geom's quaternion to a rotation matrix to find its local Z-axis (longitudinal axis)
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, model.geom_quat[primary_geom_id])
        capsule_z_axis = np.array([mat[2], mat[5], mat[8]])
        
        # Check that the primary geom's longitudinal axis is orthogonal to the joint axis
        ax = model.jnt_axis[j_id]
        if abs(np.dot(ax, capsule_z_axis)) > 1e-4:
            return False

    # 9. Check Joint Actuators (<motor>)
    motor_specs = {
        "joint0_motor": j0_id,
        "joint1_motor": j1_id,
        "joint2_motor": j2_id,
    }

    for motor_name, target_j_id in motor_specs.items():
        act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if act_id == -1: 
            return False
            
        if model.actuator_trntype[act_id] != mj.mjtTrn.mjTRN_JOINT: 
            return False
        if model.actuator_trnid[act_id, 0] != target_j_id: 
            return False
            
        expected_gear = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not np.allclose(model.actuator_gear[act_id], expected_gear, atol=1e-4): 
            return False

        if not model.actuator_ctrllimited[act_id]:
            return False
        expected_ctrlrange = np.array([-1.0, 1.0])
        if not np.allclose(model.actuator_ctrlrange[act_id], expected_ctrlrange, atol=1e-4):
            return False

        # Enforce pure motor dynamics (No <position> or <general> actuator cheats)
        if model.actuator_gaintype[act_id] != mj.mjtGain.mjGAIN_FIXED:
            return False
        if abs(model.actuator_gainprm[act_id, 0] - 1.0) > 1e-4:
            return False
            
        if model.actuator_biastype[act_id] != mj.mjtBias.mjBIAS_NONE:
            return False
        if np.any(np.abs(model.actuator_biasprm[act_id]) > 1e-4):
            return False

    # 10. Anti-cheat
    arm_joint_actuators = sum(
        1 for i in range(model.nu) 
        if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_JOINT 
        and model.actuator_trnid[i, 0] in manipulator_joints
    )
    if arm_joint_actuators != 3: 
        return False

    return True




def _check_sensors(model, spec):
    """
    Verifies Sensor requirements:
    - Total sensor count is exactly 3.
    - Includes exactly 1 accelerometer, 1 gyro, and 1 framepos sensor targeting a site.
    - All sensors are mounted on the same <site> attached to the drone body.
    - Site geometry is type 'box'.
    - Site total vertical height <= spec['sensor_site_height_max'].
    - Site bottom face rests directly on the top face of the 'central_hub' with zero clearance.
    """
    mj = _mujoco()
    
    # 1. Check Sensor Count
    if model.nsensor != 3:
        return False
        
    # 2. Gather sensor types and targeted site IDs
    sensor_types = [int(model.sensor_type[i]) for i in range(3)]
    sensor_objtypes = [model.sensor_objtype[i] for i in range(3)]
    target_site_ids = [model.sensor_objid[i] for i in range(3)]
    
    # All sensors must target a site
    if any(ot != mj.mjtObj.mjOBJ_SITE for ot in sensor_objtypes):
        return False
        
    # All sensors must target the exact same site
    if len(set(target_site_ids)) != 1:
        return False
        
    site_id = target_site_ids[0]
    
    # Verify exact required sensor set (1 Accelerometer, 1 Gyro, 1 Framepos)
    expected_types = sorted([
        int(mj.mjtSensor.mjSENS_ACCELEROMETER),
        int(mj.mjtSensor.mjSENS_GYRO),
        int(mj.mjtSensor.mjSENS_FRAMEPOS)
    ])
    if sorted(sensor_types) != expected_types:
        return False

    # 3. Check Site Parent Body (must belong to the drone body)
    drone_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
    if drone_id == -1 or model.site_bodyid[site_id] != drone_id:
        return False

    # 4. Check Site Geometry (must be box)
    if model.site_type[site_id] != mj.mjtGeom.mjGEOM_BOX:
        return False

    # 5. Check Site Height Limit (site_size[2] is half-height in Z for box)
    site_height = 2.0 * model.site_size[site_id][2]
    max_height = spec["sensor_site_height_max"]
    if site_height > max_height + 1e-4:
        return False

    # 6. Check Zero-Clearance Placement on Top of Central Hub
    hub_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "central_hub")
    if hub_id == -1:
        return False

    # Central hub top Z = hub pos Z + hub cylinder half-height
    hub_pos = model.geom_pos[hub_id]
    hub_half_height = model.geom_size[hub_id][1]
    hub_top_z = hub_pos[2] + hub_half_height

    # Site bottom Z = site pos Z - site box half-height
    site_pos = model.site_pos[site_id]
    site_half_height = model.site_size[site_id][2]
    site_bottom_z = site_pos[2] - site_half_height

    # Verify horizontal alignment (X, Y)
    if not (math.isclose(site_pos[0], hub_pos[0], abs_tol=1e-4) and 
            math.isclose(site_pos[1], hub_pos[1], abs_tol=1e-4)):
        return False

    # Verify zero clearance (bottom of site matches top of hub)
    if not math.isclose(site_bottom_z, hub_top_z, abs_tol=1e-4):
        return False

    # Validate that sensors are not spoofed with internal parameters
    for i in range(model.nsensor):
        # Check for artificial noise
        if model.sensor_noise[i] > 0.0:
            return False
        
        # Check for value clamping/cutoffs
        if model.sensor_cutoff[i] > 0.0:
            return False
        
        # Check for relative frame references (reftype 0 is mjREF_WORLD)
        # If the user targets another object to fake relative position, this catches it.
        if hasattr(model, 'sensor_reftype') and model.sensor_reftype[i] != 0:
            return False
    
    return True




def _check_mass_constraints(model, spec, xml_string=None):
    """
    Verifies Mass Constraints:
    - Total system mass is within mass['tolerance'] of mass['target'].
    - Verifies every <geom> inside <worldbody> explicitly includes the 'mass' attribute (except planes).
    - Verifies every <geom> mass value is within mass['geom_range'].
    """
    # 1. Total Mass Check (Compiled Model)
    total_mass = sum(model.body_mass)
    target_mass = spec["mass"]["target"]
    tolerance = spec["mass"]["tolerance"]
    
    if abs(total_mass - target_mass) > tolerance + 1e-4:
        return False

    # 2. Geometry Mass Explicit Attribute & Range Check (Raw XML)
    # Fail completely if we cannot verify the raw XML text
    if xml_string is None:
        return False 
        
    try:
        root = ET.fromstring(xml_string)
        min_mass, max_mass = spec["mass"]["geom_range"]

        # Iterate through every <geom> placed in the model (<worldbody> subtrees
        # only; <default> geom templates are not geoms and are skipped)
        worldbody_geoms = [g for wb in root.iter("worldbody") for g in wb.iter("geom")]
        for geom_elem in worldbody_geoms:
            
            # EXCEPTION: Skip planes entirely
            if geom_elem.get("type") == "plane":
                continue
                
            # Check if an explicit mass attribute exists
            if "mass" not in geom_elem.attrib:
                return False
                    
            # Parse the mass value
            try:
                geom_mass_val = float(geom_elem.attrib["mass"])
            except ValueError:
                return False
                
            # Check if it falls strictly within the required range
            if not (min_mass <= geom_mass_val <= max_mass):
                return False
                
    except Exception:
        # Fails if XML is malformed
        return False
            
    return True




def _check_system_dimensions(model, spec):
    """
    Checks the maximum bounding box dimensions of the settled drone.
    - Settles the drone on the ground.
    - Ensures manipulator joints stay at 0 (home position) throughout settling and measurement.
    - Calculates mathematically exact AABBs for each geometry using support mapping.
    - Verifies dimensions against spec['max_dimensions'].
    """
    mj = _mujoco()
    data = mj.MjData(model)
    
    # 1. Reset data and simulate to let the drone settle
    mj.mj_resetData(model, data)
    
    j0_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint0")
    j1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint1")
    j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint2")
    arm_jids = [j for j in [j0_id, j1_id, j2_id] if j != -1]

    # Enforce the global simulation standard
    model.opt.timestep = 0.001
    dt = model.opt.timestep
    settle_steps = int(2.5 / dt)
    
    
    for _ in range(settle_steps):
        for j_id in arm_jids:
            data.qpos[model.jnt_qposadr[j_id]] = 0.0
            data.qvel[model.jnt_dofadr[j_id]] = 0.0
            
        mj.mj_step(model, data)

    # Ensure joints are at home position and recompute forward kinematics before measuring
    for j_id in arm_jids:
        data.qpos[model.jnt_qposadr[j_id]] = 0.0
        data.qvel[model.jnt_dofadr[j_id]] = 0.0
    mj.mj_forward(model, data)

    # 2. Calculate the exact global bounding box using support mapping
    min_bounds = np.array([float('inf'), float('inf'), float('inf')])
    max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for i in range(model.ngeom):
        # Skip the worldbody (which includes the ground plane)
        if model.geom_bodyid[i] == 0:
            continue
            
        g_type = model.geom_type[i]
        size = model.geom_size[i]
        pos = data.geom_xpos[i]
        
        # 3x3 global rotation matrix of the geom
        mat = data.geom_xmat[i].reshape(3, 3)
        rbound = model.geom_rbound[i]
        
        # Calculate maximum extent along world X, Y, and Z axes
        for k in range(3):
            # The world axis k expressed in the geometry's local frame
            d_loc = mat[k, :]
            
            if g_type == mj.mjtGeom.mjGEOM_SPHERE:
                ext = size[0]
            elif g_type == mj.mjtGeom.mjGEOM_CAPSULE:
                # Radius + half-length * projection of local Z onto world axis
                ext = size[0] + size[1] * abs(d_loc[2])
            elif g_type == mj.mjtGeom.mjGEOM_CYLINDER:
                # Radius projection + half-length projection
                ext = size[1] * abs(d_loc[2]) + size[0] * np.linalg.norm(d_loc[0:2])
            elif g_type == mj.mjtGeom.mjGEOM_BOX:
                # Sum of projections of all three local axes
                ext = size[0] * abs(d_loc[0]) + size[1] * abs(d_loc[1]) + size[2] * abs(d_loc[2])
            else:
                # Fallback for complex meshes or ellipsoids
                ext = rbound 
                
            min_bounds[k] = min(min_bounds[k], pos[k] - ext)
            max_bounds[k] = max(max_bounds[k], pos[k] + ext)
            
    dim_x = max_bounds[0] - min_bounds[0]
    dim_y = max_bounds[1] - min_bounds[1]
    dim_z = max_bounds[2] - min_bounds[2]
    
    # 3. Evaluate against limits
    max_dims = spec["max_dimensions"]
    limit_x = max_dims.get("x", float('inf'))
    limit_y = max_dims.get("y", float('inf'))
    limit_z = max_dims.get("z", float('inf'))
    
    if dim_x > limit_x or dim_y > limit_y or dim_z > limit_z:
        return False
        
    
    return True




def _check_intercollision(original_model, spec, xml_string=None):
    """
    Injects virtual propellers and sweeps the manipulator through 6 kinematic 
    scenarios to check for self-collision.
    Fails closed if injection, compilation, or counts are incorrect.
    """
    mj = _mujoco()

    # 1. FAIL CLOSED IF NO XML
    if xml_string is None:
        return False
        
    # Check site limit as a backup (max 5 sites: 4 thrusters + 1 IMU)
    if original_model.nsite > 5:
        return False

    target_thrusters = {"thruster_front", "thruster_back", "thruster_left", "thruster_right"}
    propellers_injected = 0

    try:
        root = ET.fromstring(xml_string)
        
        for site_name in target_thrusters:
            site_id = mj.mj_name2id(original_model, mj.mjtObj.mjOBJ_SITE, site_name)
            if site_id == -1:
                return False # A required thruster is missing
                
            # 1. Identify which body the site belongs to in the COMPILED model
            body_id = original_model.site_bodyid[site_id]
            
            # 2. Find the corresponding XML node (<body> or <worldbody>)
            if body_id == 0:
                body_node = root.find(".//worldbody")
            else:
                body_name = mj.mj_id2name(original_model, mj.mjtObj.mjOBJ_BODY, body_id)
                if not body_name:
                    return False
                
                body_node = None
                for b in root.iter('body'):
                    if b.get('name') == body_name:
                        body_node = b
                        break
                        
            if body_node is None:
                return False
                
            # 3. Extract LOCAL coordinates relative to that body
            # This completely bypasses the <frame> exploit!
            local_pos = original_model.site_pos[site_id]
            local_quat = original_model.site_quat[site_id]
            
            pos_str = f"{local_pos[0]} {local_pos[1]} {local_pos[2]}"
            quat_str = f"{local_quat[0]} {local_quat[1]} {local_quat[2]} {local_quat[3]}"
            
            # Calculate propeller radius in meters (half of the spec diameter)
            prop_radius = spec["propellers_diameter"] / 2.0

            # 4. Create and inject the virtual propeller strictly as a child of its parent body
            geom = ET.Element('geom')
            geom.set('type', 'cylinder')
            geom.set('size', f'{prop_radius} 0.001') 
            geom.set('pos', pos_str)
            geom.set('quat', quat_str)
            geom.set('rgba', '1.0 0.0 0.0 0.5')
            
            body_node.append(geom)
            propellers_injected += 1
                        
        # 2. VERIFY EXACTLY 4 PROPELLERS CREATED
        if propellers_injected != 4:
            return False
            
        modified_xml = ET.tostring(root, encoding='unicode')
        
        # Compile new model with virtual propellers
        model = mj.MjModel.from_xml_string(modified_xml)
        
    except Exception:
        # 3. FAIL CLOSED ON ANY EXCEPTION
        return False
                
    data = mj.MjData(model)

    # --- 4. ANTI-CHEAT & FILTER BACKUP ---
    orig_contype = np.copy(model.geom_contype)
    orig_conaffinity = np.copy(model.geom_conaffinity)
    orig_disableflags = model.opt.disableflags

    try:
        model.geom_contype[:] = 1
        model.geom_conaffinity[:] = 1
        
        # Disable parent filtering so link1 <-> drone contacts are actually caught
        model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_FILTERPARENT

        # --- 5. SETUP BODY CLASSIFICATIONS ---
        link1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link1")
        link2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
        if link1_id == -1 or link2_id == -1:
            return False
            
        manip_bodies = {link1_id, link2_id}
        drone_bodies = set(range(1, model.nbody)) - manip_bodies

        # --- 6. JOINT SETUP & LIMITS FROM SPEC.JSON ---
        j0_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint0")
        j1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint1")
        j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint2")
        
        if j0_id == -1 or j1_id == -1 or j2_id == -1:
            return False
            
        q_j0, q_j1, q_j2 = model.jnt_qposadr[[j0_id, j1_id, j2_id]]
        
        j0_rng_deg = spec["joints_range"]["shoulder_vertical"]
        j0_full = (math.radians(j0_rng_deg[0]), math.radians(j0_rng_deg[1]))
        

        scenarios = [
            (j0_full, 0, math.radians(-30)),                               
            (j0_full, math.radians(90), 0),                                         
            ((math.radians(50), math.radians(310)), math.radians(-90), math.radians(90)),           
            ((math.radians(-130), math.radians(130)), math.radians(90), math.radians(90)),         
            ((math.radians(10), math.radians(350)), math.radians(-90), math.radians(-10)),
            ((math.radians(10), math.radians(350)), math.radians(-45), math.radians(-30)) 
        ]

        # Grab arm_base ID to strictly whitelist its collisions with link1
        arm_base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "arm_base")
        if arm_base_id == -1:
            return False # Failsafe if the topology check somehow missed it
            
        # --- 7. RUN KINEMATIC SWEEPS ---        
        for s_idx, (j0_rng, j1_val, j2_val) in enumerate(scenarios):
            for v_angle in np.linspace(j0_rng[0], j0_rng[1], 100):
                # Assign target joint angles to qpos before forward kinematics
                data.qpos[q_j0] = v_angle
                data.qpos[q_j1] = j1_val
                data.qpos[q_j2] = j2_val

                # Forward kinematics (computes global positions and contacts)
                mj.mj_forward(model, data)
                        
                # Get the global XYZ coordinates of the shoulder joint
                j0_global_pos = data.xanchor[j0_id]
                
                # Check all contacts in this configuration
                for i in range(data.ncon):
                    contact = data.contact[i]
                    g1, g2 = contact.geom1, contact.geom2
                    b1 = model.geom_bodyid[g1]
                    b2 = model.geom_bodyid[g2]
                    
                    # Helper: Is this body/geom an obstacle?
                    def is_drone_obstacle(b_id, g_id):
                        if b_id in drone_bodies:
                            return True
                        # As requested by the QA linter: Include ALL non-plane body 0 geoms
                        if b_id == 0 and model.geom_type[g_id] != mj.mjtGeom.mjGEOM_PLANE:
                            return True
                        return False
                    
                    # If a collision bridges a manipulator body and a drone/world obstacle
                    if (b1 in manip_bodies and is_drone_obstacle(b2, g2)) or \
                       (b2 in manip_bodies and is_drone_obstacle(b1, g1)):
                        
                        # --- STRICT HINGE EXEMPTION (Fixing the Bracket Exploit) ---
                        # 1. Verify the exact colliding pair is [arm_base] and [a geometry from link1]
                        is_base_contact = (g1 == arm_base_id and model.geom_bodyid[g2] == link1_id) or \
                                          (g2 == arm_base_id and model.geom_bodyid[g1] == link1_id)
                        
                        if is_base_contact:
                            # 2. Verify spatial distance to ensure they are only touching at the physical hinge
                            contact_pos = contact.pos
                            dist_to_hinge = np.linalg.norm(contact_pos - j0_global_pos)
                            
                            if dist_to_hinge < 0.02:
                                continue # Safe! Only the designated mount is allowed to overlap here
                                
                        # If it is ANY OTHER geometry OR outside the hinge zone, FAIL
                        return False                        
        return True

    finally:
        # RESTORE COLLISION MASKS AND FLAGS
        model.geom_contype[:] = orig_contype
        model.geom_conaffinity[:] = orig_conaffinity
        model.opt.disableflags = orig_disableflags





def _check_ee_clearance(model, spec):
    """
    Verifies End Effector Clearance:
    - Simulates the drone dropping to the ground and settling.
    - Leaves manipulator joints unforced and free during the 5 s settle (no position/velocity overrides).
    - Finds the end point (EE) of link2's primary capsule.
    - Checks vertical distance from EE to central_hub >= to_central_hub.
    - Checks vertical distance from EE to ground >= to_ground.
    """
    mj = _mujoco()
    data = mj.MjData(model)
    

    # 1. Reset data and simulate for 5.0 seconds to let the drone and unforced joints settle naturally on the ground
    mj.mj_resetData(model, data)
    
    j2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm_joint2")

    # FIX: Clamp the timestep so agents cannot force billions of steps
    model.opt.timestep = 0.001
    dt = model.opt.timestep
    max_steps = int(5.0 / dt)
    
    for _ in range(max_steps):
        mj.mj_step(model, data)
        
        # Bail out early if physics explode (NaN detected)
        if not np.all(np.isfinite(data.qpos)):
            return False

    # Recompute forward kinematics for the settled pose before measuring
    mj.mj_forward(model, data)


    # 2. Get the Central Hub's Z position
    hub_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "central_hub")
    if hub_id == -1: 
        return False
    hub_z = data.geom_xpos[hub_id][2]
    
    # 3. Get the Ground Plane's Z position
    plane_z = 0.0
    for i in range(model.ngeom):
        if model.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE:
            plane_z = data.geom_xpos[i][2]
            break

    # 4. Find link2's primary capsule and calculate its far end (End Effector)
    link2_b_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
    if link2_b_id == -1: 
        return False
        
    gids_link2 = [g for g in range(model.ngeom) if model.geom_bodyid[g] == link2_b_id]
    capsules_l2 = [g for g in gids_link2 if model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE]
    if not capsules_l2: 
        return False
        
    primary_l2 = max(capsules_l2, key=lambda g: model.geom_size[g][1])
    
    # Get global position (center) and rotation matrix of the capsule
    c = data.geom_xpos[primary_l2]
    mat = data.geom_xmat[primary_l2].reshape(3, 3)
    z_axis = mat[:, 2] # The local Z-axis vector in global coordinates
    
    # A capsule extends along its local Z-axis by (half_length + radius) including hemispherical caps
    radius = model.geom_size[primary_l2][0]
    half_len = model.geom_size[primary_l2][1]
    total_extent = half_len + radius
    
    end1 = c + (z_axis * total_extent)
    end2 = c - (z_axis * total_extent)
    
    # The end effector is the end FURTHEST from joint2 / link2 attachment
    ref_pos = data.xanchor[j2_id] if j2_id != -1 else data.xpos[link2_b_id]
    dist1 = np.linalg.norm(end1 - ref_pos)
    dist2 = np.linalg.norm(end2 - ref_pos)
    ee_pos = end1 if dist1 >= dist2 else end2
    
    ee_z = ee_pos[2]

    # 5. Evaluate Clearances
    clearance_spec = spec.get("ee_clearance", spec.get("ee_clearance_min", spec))
    if not isinstance(clearance_spec, dict):
        clearance_spec = {}

    min_to_hub = clearance_spec.get("to_central_hub", clearance_spec.get("central_hub", spec.get("to_central_hub", 0.15)))
    min_to_ground = clearance_spec.get("to_ground", clearance_spec.get("ground", spec.get("to_ground", 0.10)))
    
    # Absolute vertical distance between hub and EE
    dist_to_hub = abs(hub_z - ee_z)
    # Vertical distance from EE to the ground plane
    dist_to_ground = ee_z - plane_z
    
    if dist_to_hub < min_to_hub - 1e-4:
        return False
        
    if dist_to_ground < min_to_ground - 1e-4:
        return False

    return True

    


def _check_push_stability(model, spec):
    """
    Checks the push stability of the drone without permanently mutating the model.
    """
    mj = _mujoco()
    
    # 1. Backup the original flags and disable actuators
    original_flags = model.opt.disableflags
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_ACTUATION

    try:
        data = mj.MjData(model)
        # 1. Enforce the global simulation standard
        model.opt.timestep = 0.001
        dt = model.opt.timestep

        # Load parameters from spec
        push_cfg = spec["push_test"]
        force_magnitude = push_cfg["force_n"]
        duration_sec = push_cfg["duration_sec"]
        min_required_tilt, max_allowed_tilt = push_cfg["tilt_range_deg"]
        max_displacement = push_cfg["displacement_max"]
        max_yaw_change = push_cfg["yaw_change_max"]

        settle_steps = int(3.0 / dt)
        push_steps = int(duration_sec / dt)
        observe_steps = int(4.0 / dt)

        MAX_ALLOWED_STEPS = 100_000
        if (settle_steps + push_steps + observe_steps) > MAX_ALLOWED_STEPS:
            return False

        # Find central_hub geom and parent body
        hub_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "central_hub")
        if hub_id == -1:
            print("Error: central_hub geometry not found.")
            return False
            
        drone_body_id = model.geom_bodyid[hub_id]

        def _get_z_axis():
            return data.xmat[drone_body_id].reshape(3, 3)[:, 2].copy()
            
        def _get_yaw():
            mat = data.xmat[drone_body_id].reshape(3, 3)
            return math.atan2(mat[1, 0], mat[0, 0])
            
        
        # --- Phase 1: Wait for landing and settling ---
        for _ in range(settle_steps):
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False

        baseline_z_axis = _get_z_axis()
        baseline_yaw = _get_yaw()
        baseline_xy = data.xpos[drone_body_id][:2].copy()

        def _get_tilt():
            curr_z_axis = _get_z_axis()
            dot_prod = np.clip(np.dot(baseline_z_axis, curr_z_axis), -1.0, 1.0)
            return math.degrees(math.acos(dot_prod))

        max_tilt_achieved = 0.0
        force_vec = np.array([force_magnitude, 0.0, 0.0])

        # --- Phase 2: Apply push force at central_hub location ---
        for _ in range(push_steps):
                
            p_hub = data.geom_xpos[hub_id]
            p_com = data.xipos[drone_body_id]
            r = p_hub - p_com
            torque_vec = np.cross(r, force_vec)

            data.xfrc_applied[drone_body_id, :3] = force_vec
            data.xfrc_applied[drone_body_id, 3:] = torque_vec

            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False

            curr_tilt = _get_tilt()
            max_tilt_achieved = max(max_tilt_achieved, curr_tilt)

            if curr_tilt > max_allowed_tilt:
                return False

        data.xfrc_applied[drone_body_id, :] = 0.0

        # --- Phase 3: Observe reaction ---
        for _ in range(observe_steps):
                
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False
            
            curr_tilt = _get_tilt()
            max_tilt_achieved = max(max_tilt_achieved, curr_tilt)
            if curr_tilt > max_allowed_tilt:
                return False

        # --- Final State Checks ---
        if max_tilt_achieved < min_required_tilt:
            return False
            
        final_xy = data.xpos[drone_body_id][:2]
        displacement = np.linalg.norm(final_xy - baseline_xy)
        if displacement > max_displacement:
            return False
            
        final_yaw = _get_yaw()
        yaw_diff_rad = (final_yaw - baseline_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_diff_deg = abs(math.degrees(yaw_diff_rad))
        if yaw_diff_deg > max_yaw_change:
            return False

        return True

    finally:
        # This guarantees motors are re-enabled for the flight test!
        model.opt.disableflags = original_flags




_DECLARED_YAW_COEFF = {"thruster_front": 0.1, "thruster_back": 0.1, "thruster_left": -0.1, "thruster_right": -0.1}


def _body_yaw(data, body_id):
    m = data.xmat[body_id]
    return math.atan2(m[3], m[0])


def _body_tilt_deg(data, body_id):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body_id][8])))))


class _FlightController:
    """Grader-supplied cascaded flight controller (deterministic, public in instruction.md).

    XY position PD -> desired roll/pitch (clipped) ; altitude PD + tilt-compensated gravity
    feed-forward -> collective ; roll/pitch/yaw PD on body rates -> torques ; 4x4 mixer built from
    the model's thruster site positions (drone frame) and the DECLARED yaw coefficients.
    """

    def __init__(self, model, drone_body_id, rotor_actuator_ids, cfg):
        mj = _mujoco()
        self.cfg = cfg
        self.did = drone_body_id
        self.rot = list(rotor_actuator_ids)
        self.mass = mj.mj_getTotalmass(model)
        self.dof = int(model.jnt_dofadr[model.body_jntadr[drone_body_id]])
        self.saturated_steps = 0
        A = np.zeros((4, 4))
        self.ok = True
        for k, a in enumerate(self.rot):
            s_id = model.actuator_trnid[a, 0]
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, s_id)
            if name not in _DECLARED_YAW_COEFF or model.site_bodyid[s_id] != drone_body_id:
                self.ok = False
                return
            p = model.site_pos[s_id]
            A[:, k] = [1.0, p[1], -p[0], _DECLARED_YAW_COEFF[name]]
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
        w = data.qvel[self.dof + 3:self.dof + 6]  # body-frame angular velocity (gyro)
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


def evaluate_flight_test(model, spec):
    """
    Evaluates the drone model against the two flight test missions defined in spec.json.
    Returns a tuple (Passed: bool, Reason: str).
    """
    
    mj = _mujoco()
    data = mj.MjData(model)

    # --- DYNAMIC ROTOR LOOKUP ---
    target_sites = {"thruster_front", "thruster_back", "thruster_left", "thruster_right"}
    target_site_ids = set()
    
    # Get the internal IDs for our specific thruster sites
    for site_name in target_sites:
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            target_site_ids.add(site_id)
            
    rotor_actuator_ids = []
    
    # Loop through all actuators in the model
    for i in range(model.nu):
        # Check if the actuator is attached to a site (mjTRN_SITE)
        if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_SITE:
            # Check if that site is one of our thrusters
            site_id = model.actuator_trnid[i, 0]
            if site_id in target_site_ids:
                rotor_actuator_ids.append(i)
                
    # Failsafe: If we didn't find exactly 4 rotor actuators, fail the flight test
    if len(rotor_actuator_ids) != 4:
        return False
    
    ft_spec = spec["flight_test"]

    # Enforce the global simulation standard
    model.opt.timestep = 0.001
    dt = model.opt.timestep

    drone_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
    
    # Extract controller parameters
    kp_arm, ki_arm, kd_arm = ft_spec["pid_arm"]["Kp"], ft_spec["pid_arm"]["Ki"], ft_spec["pid_arm"]["Kd"]
    
    target_z_disp = ft_spec["target_z_displacement"]
    max_xy_drift_target = ft_spec["max_xy_drift"]["target"]
    max_xy_drift_tol = ft_spec["max_xy_drift"]["tolerance"]

    # IDs and indices
    joint_names = ["arm_joint0", "arm_joint1", "arm_joint2"]
    motor_names = ["joint0_motor", "joint1_motor", "joint2_motor"]
    
    try:
        joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name) for name in motor_names]
        qpos_idxs = [model.jnt_qposadr[jid] for jid in joint_ids]
        qvel_idxs = [model.jnt_dofadr[jid] for jid in joint_ids]
    except Exception:
        return False
    

    # Define the test profiles based on spec. 
    # Use None for joints that should NOT be controlled during a stage.
    test_profiles = [
        {
            "name": "Test 1",
            "final_targets": [
                math.radians(ft_spec["test_1"]["joint0_target"]),
                math.radians(ft_spec["test_1"]["joint1_target"]),
                math.radians(ft_spec["test_1"]["joint2_target"])
            ],
            "stages": [
                {
                    "duration": ft_spec["arm_target_duration_1"],
                    "targets": [
                        math.radians(ft_spec["test_1"]["joint0_target"]),
                        math.radians(ft_spec["test_1"]["joint1_target"]),
                        math.radians(ft_spec["test_1"]["joint2_target"])
                    ]
                }
            ]
        },
        {
            "name": "Test 2",
            "final_targets": [
                math.radians(ft_spec["test_2"]["joint0_target_2"]),
                math.radians(ft_spec["test_2"]["joint1_target_2"]),
                math.radians(ft_spec["test_2"]["joint2_target_2"])
            ],
            "stages": [
                {
                    "duration": ft_spec["arm_target_duration_1"],
                    "targets": [
                        None,
                        math.radians(ft_spec["test_2"]["joint1_target_1"]),
                        None
                    ]
                },
                {
                    "duration": ft_spec["arm_target_duration_2"],
                    "targets": [
                        math.radians(ft_spec["test_2"]["joint0_target_2"]),
                        math.radians(ft_spec["test_2"]["joint1_target_2"]),
                        math.radians(ft_spec["test_2"]["joint2_target_2"])
                    ]
                }
            ]
        }
    ]

    for test in test_profiles:
        mj.mj_resetData(model, data)
        integral_errors = [0.0, 0.0, 0.0]

        # --- Phase 1: Settle ---
        settle_steps = int(ft_spec["settle_time"] / dt)
        for _ in range(settle_steps):
            for act_id in actuator_ids:
                data.ctrl[act_id] = 0.0
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False

        # Record baseline coordinates (Replacement 1)
        
        initial_pos = data.xpos[drone_body_id].copy()
        z_0 = initial_pos[2]

        def run_arm_deployment(duration, targets):
            steps = int(duration / dt)
            for _ in range(steps):
                for j in range(3):
                    if targets[j] is not None:
                        pos_err = targets[j] - data.qpos[qpos_idxs[j]]
                        vel_err = 0.0 - data.qvel[qvel_idxs[j]]
                        integral_errors[j] += pos_err * dt
                        u = (kp_arm * pos_err) + (ki_arm * integral_errors[j]) + (kd_arm * vel_err)
                        data.ctrl[actuator_ids[j]] = max(-1.0, min(1.0, u))
                mj.mj_step(model, data)
                if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                    return False
            
            # Verify joint convergence at end of stage (tolerance of 5 degrees)
            for j in range(3):
                if targets[j] is not None:
                    if abs(targets[j] - data.qpos[qpos_idxs[j]]) > math.radians(5.0):
                        return False
            return True

        # --- Phase 2: Arm Deployment Stages ---
        for stage in test["stages"]:
            if not run_arm_deployment(stage["duration"], stage["targets"]):
                return False
        
        # --- Phase 3: Takeoff & Hover (grader-supplied cascaded flight controller) ---
        ctl = _FlightController(model, drone_body_id, rotor_actuator_ids, ft_spec["controller"])
        if not ctl.ok:
            return False
        hover_steps = int(ft_spec["hover_time"] / dt)
        target_hover_z = z_0 + target_z_disp
        xy_0 = initial_pos[:2].copy()
        yaw_0 = _body_yaw(data, drone_body_id)
        yaw_1 = yaw_0 + math.radians(ft_spec["controller"]["yaw_step_deg"])
        max_z_reached = -float('inf')
        max_tilt_deg = 0.0

        def hold_arm():
            for j in range(3):
                pos_err = test["final_targets"][j] - data.qpos[qpos_idxs[j]]
                vel_err = 0.0 - data.qvel[qvel_idxs[j]]
                integral_errors[j] += pos_err * dt
                u = (kp_arm * pos_err) + (ki_arm * integral_errors[j]) + (kd_arm * vel_err)
                data.ctrl[actuator_ids[j]] = max(-1.0, min(1.0, u))

        for step in range(hover_steps):
            hold_arm()
            # first half: hold heading; second half: commanded heading step (yaw authority check)
            yaw_cmd = yaw_0 if step < hover_steps // 2 else yaw_1
            ctl.command(data, target_hover_z, xy_0, yaw_cmd, log=step >= hover_steps // 2)
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False
            max_z_reached = max(max_z_reached, data.xpos[drone_body_id][2])
            max_tilt_deg = max(max_tilt_deg, _body_tilt_deg(data, drone_body_id))

        # Verify altitude reach (tolerance of 0.2 meters)
        if abs(max_z_reached - target_hover_z) > 0.2:
            return False
        # Verify heading tracking at end of hover
        yaw_err = (_body_yaw(data, drone_body_id) - yaw_1 + math.pi) % (2 * math.pi) - math.pi
        if abs(math.degrees(yaw_err)) > ft_spec["controller"]["yaw_tolerance_deg"]:
            return False
        # Verify no rotor saturated while hovering under the commanded heading
        if ctl.saturated_steps > 0:
            return False

        # --- Phase 4: Landing ---
        landing_steps = int(ft_spec["landing_time"] / dt)
        for step in range(landing_steps):
            hold_arm()
            progress = (step + 1) / landing_steps
            current_target_z = target_hover_z + (z_0 - target_hover_z) * progress
            ctl.command(data, current_target_z, xy_0, yaw_1, log=False)
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False
            max_tilt_deg = max(max_tilt_deg, _body_tilt_deg(data, drone_body_id))

        if max_tilt_deg > ft_spec["controller"]["max_tilt_deg"]:
            return False

        # Cut motors and let settle briefly
        for r in rotor_actuator_ids:
            data.ctrl[r] = 0.0
        for _ in range(int(1.0 / dt)):
            mj.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or data.warning.number.any():
                return False

        # --- Phase 5: Final Evaluation ---
        # (Replacement 4)
        final_pos = data.xpos[drone_body_id].copy()
        drift_xy = math.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)
        z_diff = abs(final_pos[2] - z_0)

        # Verify final resting Z is near original baseline (tolerance 0.05 meters)
        if z_diff > 0.05:
            return False
        
        # Verify drift
        if drift_xy > max_xy_drift_target + max_xy_drift_tol:
            return False
        
    return True




def _check_complexity_limits(model, xml_string=None):
    """
    Verifies Complexity Limits:
    - Maximum 30 geoms.
    - Maximum 5 bodies (worldbody + up to 4 user-defined).
    - Maximum 5 joints (including freejoint).
    - Max damping of 1.0 and frictionloss of 0.5 for extra joints.
    - Exactly 7 motors (actuators).
    - Exactly 1 geom of type plane.
    - No equality constraints allowed (model.neq == 0).
    - No quat or euler attributes in the raw XML.
    - Timestep must be exactly 0.001.
    - Maximum armature of 0.0001 for all joints.
    - Exactly 5 sites.
    - Solver iterations <= 150.
    - Solver tolerance >= 1e-8.
    - All geometries must be capsule, cylinder, or plane.
    - Prevent tendons (model.ntendon == 0).
    """
    mj = _mujoco()
    
    # 1. Check Geometries
    if model.ngeom > 30:
        return False
        
    # 2. Check Bodies
    if model.nbody > 5:
        return False
        
    # 3. Check Joints and their properties
    if model.njnt > 5:
        return False
        
    # Enforce damping and frictionloss limits on all degrees of freedom (DOFs).
    # This prevents users from adding a highly damped 5th joint to game stability.
    for i in range(model.nv):
        if model.dof_damping[i] > 1.0:
            return False
        if model.dof_frictionloss[i] > 0.2:
            return False
            
    # 4. Check Motors (Actuators)
    if model.nu != 7:
        return False
        
    # --- 5. Check Allowed Types, Ground Plane & Worldbody Scenery ---
    plane_count = 0
    worldbody_geom_count = 0
    
    for i in range(model.ngeom):
        g_type = model.geom_type[i]
        
        # Get the name of the geometry (returns None if unnamed)
        g_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        
        if g_type == mj.mjtGeom.mjGEOM_PLANE:
            plane_count += 1
        elif g_type == mj.mjtGeom.mjGEOM_CYLINDER:
            # Only the central_hub is allowed to be a cylinder
            if g_name != "central_hub":
                return False
        elif g_type == mj.mjtGeom.mjGEOM_CAPSULE:
            # Capsules are the standard allowed type
            pass
        else:
            # Instantly fail if an illegal geometry (like a box or sphere) is used
            return False
            
        # Static scenery: any geom welded to the world (worldbody geoms and
        # geoms of jointless bodies under worldbody) counts as scenery.
        if model.body_weldid[model.geom_bodyid[i]] == 0:
            worldbody_geom_count += 1
            if g_type != mj.mjtGeom.mjGEOM_PLANE:
                return False

    if plane_count != 1 or worldbody_geom_count != 1:
        return False

    # Every user body must belong to the drone's kinematic tree (no other
    # world-attached or free bodies anywhere in the environment).
    drone_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "drone")
    if drone_id == -1:
        return False
    for b in range(1, model.nbody):
        if model.body_rootid[b] != drone_id:
            return False

    # Actuators must be pure motors: no activation dynamics, fixed unit-free
    # gain, no bias (rules out <general>/<position>/<velocity>/filter tricks).
    for i in range(model.nu):
        if model.actuator_dyntype[i] != mj.mjtDyn.mjDYN_NONE:
            return False
        if model.actuator_gaintype[i] != mj.mjtGain.mjGAIN_FIXED:
            return False
        if model.actuator_biastype[i] != mj.mjtBias.mjBIAS_NONE:
            return False

    # Compiled capsule geometry (covers fromto, pos/size + axis, class defaults):
    # both segment endpoints, in the owning body's frame, on the 0.01 m grid,
    # and radius >= 0.005 m.
    for i in range(model.ngeom):
        if model.geom_type[i] != mj.mjtGeom.mjGEOM_CAPSULE:
            continue
        if model.geom_size[i][0] < 0.005 - 1e-5:
            return False
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, model.geom_quat[i])
        z_axis = mat.reshape(3, 3)[:, 2]
        half_len = model.geom_size[i][1]
        for end in (model.geom_pos[i] + z_axis * half_len, model.geom_pos[i] - z_axis * half_len):
            for val in end:
                if not math.isclose(val, round(val / 0.01) * 0.01, abs_tol=1e-5):
                    return False
    
    # 6. Check Equality Constraints
    if model.neq > 0:
        return False
        
    # 7. Check Timestep
    if not math.isclose(model.opt.timestep, 0.001, abs_tol=1e-5):
        return False

    # 8. Check Sites Limit
    if model.nsite != 5:
        return False
        
    # 9. Check Solver Settings
    if model.opt.iterations > 150:
        return False
        
    if model.opt.tolerance < 1e-8:
        return False

    # --- 10. Engineering Standardization (Resolutions and Limits) ---
    # Fail closed if raw XML is missing, as we cannot verify original text inputs
    if xml_string is None:
        return False 
        
    try:
        root = ET.fromstring(xml_string)

        # NEW: Ensure exactly one worldbody exists to prevent hidden geometry exploits
        if len(root.findall(".//worldbody")) > 1:
            return False

        # Every element inside an <actuator> section must be a <motor>.
        for act_section in root.iter("actuator"):
            for child in act_section:
                if isinstance(child.tag, str) and child.tag != "motor":
                    return False

        # UPDATED: Ban 'quat' and 'euler' attributes, and ban 'size'/'flag' tags entirely
        for elem in root.iter():
            if "quat" in elem.attrib or "euler" in elem.attrib:
                return False
            if elem.tag in ["size", "flag"]:
                return False
        
        # Check every geometry in the raw XML (this already iterates correctly over defaults)
        for geom_elem in root.iter("geom"):
            g_type = geom_elem.get("type", "sphere") # Get type, default to sphere if omitted
            
            if g_type == "capsule":
                # 1. Capsule 'fromto' resolution (must be 0.01 increments)
                if "fromto" in geom_elem.attrib:
                    fromto_vals = [float(x) for x in geom_elem.attrib["fromto"].split()]
                    for val in fromto_vals:
                        # Find the closest 0.01 increment and compare
                        expected_val = round(val / 0.01) * 0.01
                        if not math.isclose(val, expected_val, abs_tol=1e-5):
                            return False
                            
                # 2. Capsule 'size' minimum (must be >= 0.005)
                if "size" in geom_elem.attrib:
                    size_vals = [float(x) for x in geom_elem.attrib["size"].split()]
                    for val in size_vals:
                        if val < 0.005 - 1e-5:
                            return False
                            
            # 3. All geometries 'mass' resolution (must be 0.002 increments)
            if "mass" in geom_elem.attrib:
                mass_val = float(geom_elem.attrib["mass"])
                expected_mass = round(mass_val / 0.002) * 0.002
                if not math.isclose(mass_val, expected_mass, abs_tol=1e-5):
                    return False
                    
    except Exception:
        # Fails closed if the XML is malformed or attributes aren't valid floats
        return False

    # --- 11. Check Joint Armature Limit ---
    # Enforce that no DOF in the system exceeds an armature of 0.0001
    for i in range(model.nv):
        if model.dof_armature[i] > 0.0001:
            return False

    # --- 14. PREVENT TENDONS ---
    # Prevents agents from attaching hidden stiffness, damping, or friction to joints via tendons
    if model.ntendon > 0:
        return False
      
    return True




def _check_honest_physics(model, xml_string=None):
    """
    Verifies the agent hasn't tampered with gravity, collisions, or contact rules:
    - 1. The free joint (flight DOFs) must not have damping, armature, or frictionloss.
    - 2. Gravity must be Earth-like (~[0, 0, -9.81]).
    - 3. A ground plane must exist on the worldbody (Body 0).
    - 4. Global contacts must be enabled, and enableflags must be 0 (no overrides).
    - 5. No geometries can have contype=0, conaffinity=0, gap>0, or margin>0.
    - 6. No geometry name can contain "propeller".
    - 7. Sliding friction <= 1.5, and priority/solref/solimp overrides are forbidden.
    - 8. No contact exclusions (nexclude) are allowed.
    - 9. Global solver options (cone, impratio, noslip_iterations, integrator, aerodynamics) must remain default.
    - 10. No body gravity compensation (gravcomp == 0).
    - 11. No joint stiffness on any joint.
    - 12. No explicit <inertial> elements overriding natural geometry inertia, and no <compiler> overrides.
    - 13. No explicit contact pairs (npair == 0) to prevent margin/gap/friction overrides.
    - 14. No banned structural tags (<include>, <replicate>, <composite>, <flexcomp>).
    - 15. The <size memory="X"> attribute cannot exceed 100MB.
    """
    import numpy as np
    mj = _mujoco()
    
    # --- 1. FREE JOINT PHYSICS CHECK ---
    if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
        qvel_start = model.jnt_dofadr[0]
        for i in range(6):
            dof_idx = qvel_start + i
            if model.dof_damping[dof_idx] > 0.0 or model.dof_armature[dof_idx] > 0.0 or model.dof_frictionloss[dof_idx] > 0.0:
                return False

    # --- 2. CHECK GRAVITY ---
    if abs(model.opt.gravity[2] - (-9.81)) > 0.5: return False
    if abs(model.opt.gravity[0]) > 0.1: return False
    if abs(model.opt.gravity[1]) > 0.1: return False
        
    # --- 3. CHECK FOR GROUND PLANE ---
    has_floor = False
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0 and int(model.geom_type[i]) == mj.mjtGeom.mjGEOM_PLANE:
            has_floor = True
            break
    if not has_floor: return False
        
    # --- 4. CHECK GLOBAL COLLISION ENABLEMENT ---
    if model.opt.disableflags & mj.mjtDisableBit.mjDSBL_CONTACT:
        return False
        
    if model.opt.enableflags != 0:
        return False
        
    # --- 5, 6, & 7. GEOMETRY ATTRIBUTES CHECK ---
    default_solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
    for i in range(model.ngeom):
        if model.geom_contype[i] != 1 or model.geom_conaffinity[i] != 1:
            return False
            
        # Check sliding friction AND torsional/rolling friction bounds
        if model.geom_friction[i][0] > 1.5:
            return False
        if model.geom_friction[i][1] > 0.01 or model.geom_friction[i][2] > 0.01:
            return False
            
        # Ensure contact dimensionality doesn't enable unnatural friction
        if model.geom_condim[i] != 3:
            return False

        # Geoms must stay in the default group 0 (group-dependent compiler
        # options such as inertiagrouprange could otherwise drop their inertia).
        if model.geom_group[i] != 0:
            return False
            
        if model.geom_priority[i] != 0:
            return False

        if abs(model.geom_solref[i][0] - 0.02) > 1e-3 or abs(model.geom_solref[i][1] - 1.0) > 1e-3:
            return False
            
        if not np.allclose(model.geom_solimp[i], default_solimp, atol=1e-3):
            return False
            
        if abs(model.geom_margin[i]) > 1e-4 or abs(model.geom_gap[i]) > 1e-4:
            return False
            
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "propeller" in geom_name.lower():
            return False
            
    # --- 8. PREVENT CONTACT EXCLUSIONS ---
    if model.nexclude > 0:
        return False
        
    # --- 9. CHECK GLOBAL SOLVER OPTIONS & AERODYNAMICS ---
    if model.opt.cone != mj.mjtCone.mjCONE_PYRAMIDAL: return False
    if abs(model.opt.impratio - 1.0) > 1e-3: return False
    if model.opt.noslip_iterations > 0: return False
    
    if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER: 
        return False

    if abs(model.opt.density) > 1e-4 or abs(model.opt.viscosity) > 1e-4: 
        return False
    if np.any(np.abs(model.opt.wind) > 1e-4): 
        return False

    # --- 10. PREVENT BODY GRAVITY COMPENSATION ---
    if np.any(np.abs(model.body_gravcomp) > 1e-4):
        return False

    # --- 11. PREVENT JOINT STIFFNESS OVERRIDES ---
    if np.any(np.abs(model.jnt_stiffness) > 1e-4):
        return False

    # --- 12. PREVENT MANUAL <inertial> AND <compiler> OVERRIDES IN XML ---
    if xml_string is not None:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_string)
            
            # Reject explicit inertial tags
            if root.find(".//inertial") is not None:
                return False
                
            # Reject compiler physics overrides
            banned_compiler_attrs = {
                "settotalmass", "boundmass", "boundinertia", "balanceinertia",
                "inertiagrouprange", "inertiafromgeom", "fusestatic", "alignfree",
            }
            for compiler_tag in root.findall(".//compiler"):
                if banned_compiler_attrs.intersection(compiler_tag.attrib.keys()):
                    return False
                
            # Reject banned structural macro tags
            banned_tags = {"include", "replicate", "composite", "flexcomp"}
            for tag in banned_tags:
                if root.find(f".//{tag}") is not None:
                    return False

            # NEW: Check <option> tag for unauthorized solver attributes
            for option_elem in root.findall(".//option"):
                allowed_opts = {
                    "timestep", "iterations", "tolerance", "gravity", "cone", 
                    "impratio", "noslip_iterations", "integrator", "density", 
                    "viscosity", "wind"
                }
                for attr in option_elem.attrib:
                    if attr not in allowed_opts:
                        return False

        except Exception:
            return False
        
    # --- 13. PREVENT EXPLICIT CONTACT PAIRS ---
    # This prevents overriding default geometry constraints with invisible margins or super-friction
    if model.npair > 0:
        return False

    return True



def _rollout_is_finite(model):
    """Runs a short simulation to ensure the system is physically stable and yields no NaNs."""
    mj = _mujoco()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    steps = int(ROLLOUT_DURATION_SEC / model.opt.timestep)
    try:
        if model is None:
            return False
    except Exception:
        return False

    for _ in range(steps):
        mj.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            return False
    return True


def evaluate(context):
    def safe_bool(fn):
        try:
            return bool(fn())
        except Exception:
            return False

    spec = _load_spec()
    if not _precompile_gate(context.candidate):
        return _all_failed()
    model = context.candidate_operation("MJCF compilation", _load_model, context.candidate)
    if not _postcompile_gate(model):
        return _all_failed()

    # 1. Sanitize solver options immediately after compilation to neutralize solver bombs
    if hasattr(model, "opt"):
        model.opt.iterations = min(model.opt.iterations, 500)
        if model.opt.tolerance <= 0:
            model.opt.tolerance = 1e-8

    # 2. RUN STATIC COMPLEXITY & HONEST PHYSICS CHECKS IMMEDIATELY
    complexity_ok = safe_bool(
        lambda: _check_complexity_limits(model, xml_string=context.candidate)
    )
    honest_physics_ok = safe_bool(
        lambda: _check_honest_physics(model, xml_string=context.candidate)
    )

    # Physics-outcome criteria (settled dimensions, collision sweeps, passive
    # settling, push, flight) are only meaningful when the model's physics is
    # honest and inside the complexity limits (gravity, contacts, equality
    # constraints, tendons, solver settings). They therefore depend on both
    # static checks; every other criterion is evaluated on its own.
    physics_valid = complexity_ok and honest_physics_ok

    rollout_ok = safe_bool(
        lambda: context.candidate_operation("MuJoCo rollout", _rollout_is_finite, model)
    )

    base_frame_ok = safe_bool(lambda: _check_base_and_frame_structure(model, spec))
    contiguity_ok = safe_bool(lambda: _check_drone_contiguity(model))
    rotors_ok = safe_bool(lambda: _check_rotors_and_propellers(model, spec))
    manipulator_topology_ok = safe_bool(lambda: _check_manipulator_topology(model, spec))
    manipulator_joints_ok = safe_bool(lambda: _check_manipulator_joints(model, spec))
    sensors_ok = safe_bool(lambda: _check_sensors(model, spec))
    mass_ok = safe_bool(
        lambda: _check_mass_constraints(model, spec, xml_string=context.candidate)
    )

    dimensions_ok = physics_valid and safe_bool(lambda: _check_system_dimensions(model, spec))
    ee_clearance_ok = physics_valid and safe_bool(lambda: _check_ee_clearance(model, spec))
    intercollision_ok = physics_valid and safe_bool(
        lambda: _check_intercollision(model, spec, xml_string=context.candidate)
    )
    push_stability_ok = physics_valid and safe_bool(lambda: _check_push_stability(model, spec))
    flight_test_ok = physics_valid and safe_bool(lambda: evaluate_flight_test(model, spec))

    return {
        "compiled": 1.0,
        "honest_physics": honest_physics_ok,
        "no_nan": rollout_ok,
        "complexity_limits": complexity_ok,
        "base_frame": base_frame_ok,
        "contiguity": contiguity_ok,
        "rotors_propellers": rotors_ok,
        "manipulator_topology": manipulator_topology_ok,
        "manipulator_joints": manipulator_joints_ok,
        "sensors": sensors_ok,
        "mass_score": mass_ok,
        "system_dimensions": dimensions_ok,
        "intercollision": intercollision_ok,
        "ee_clearance": ee_clearance_ok,
        "push_stability": push_stability_ok,
        "flight_test": flight_test_ok,
    }


_CRITERION_NAMES = (
    "compiled", "no_nan", "honest_physics", "complexity_limits", "base_frame",
    "contiguity", "rotors_propellers", "manipulator_topology",
    "manipulator_joints", "sensors", "mass_score", "system_dimensions",
    "intercollision", "ee_clearance", "push_stability", "flight_test",
)

TASK = RubricTask(
    artifact=TextArtifact("model.xml", max_bytes=4 * 1024 * 1024),
    criteria=(
        RubricCriterion("compiled", weight=0.05, description="MJCF parses and compiles into a valid MuJoCo model", required=True),
        RubricCriterion("no_nan", weight=0.05, description="Physics simulation executes cleanly without NaNs", required=True),
        RubricCriterion("honest_physics", weight=0.06, description="Honest physics: Earth gravity, undamped freejoint, all contacts enabled, default condim 3/group 0/solref/solimp/margin/gap, friction bounds, no exclusions/pairs/gravcomp/stiffness/inertial or banned compiler/option overrides"),
        RubricCriterion("complexity_limits", weight=0.06, description="Complexity limits: geom/body/joint/site/actuator counts, exactly 7 pure <motor> actuators, only the ground plane as scenery and every body in the drone tree, timestep/solver bounds, no tendons/equalities/quat/euler, compiled capsules on the 0.01 m grid with radius >= 0.005 and masses on the 0.002 kg grid"),
        RubricCriterion("base_frame", weight=0.06, description="Base body named drone at pos [0,0,1] with freejoint, central hub, and four symmetric arms"),
        RubricCriterion("contiguity", weight=0.08, description="Every geom welded to the drone (including jointless child bodies) and of any extra jointed drone child body touches a chain back to central_hub at the default pose"),
        RubricCriterion("rotors_propellers", weight=0.10, description="Four vertical (tilt <= ~0.08 deg) thruster sites on the drone arms, each driven by one pure motor with the exact gear vector, with propeller-diameter spacing"),
        RubricCriterion("manipulator_topology", weight=0.10, description="Two-link series manipulator (link1 child of drone, link2 only child of link1, no bodies below link2) with capsule geoms, length, attachment and minimum link mass constraints"),
        RubricCriterion("manipulator_joints", weight=0.10, description="Three actuated hinge joints with correct ranges, axes, and bounded damping and frictionloss"),
        RubricCriterion("sensors", weight=0.06, description="Three sensors (accelerometer, gyro, framepos) mounted on a box site with zero clearance on central hub"),
        RubricCriterion("mass_score", weight=0.06, description="Total system mass within tolerance of the target and every non-plane worldbody geom carries an explicit mass inside the per-geom range"),
        RubricCriterion("system_dimensions", weight=0.06, description="Settled bounding box dimensions comply with maximum X, Y, Z limits (requires honest physics and complexity limits)"),
        RubricCriterion("intercollision", weight=0.3, description="No self-collision or collision with virtual propeller discs across kinematic sweeps (requires honest physics and complexity limits)"),
        RubricCriterion("ee_clearance", weight=0.2, description="End effector maintains required vertical clearances to hub and ground after passive settling with free joints (requires honest physics and complexity limits)"),
        RubricCriterion("push_stability", weight=0.4, description="Drone exhibits required tilt response under external horizontal push force (requires honest physics and complexity limits)"),
        RubricCriterion("flight_test", weight=2.7, description="Drone successfully performs arm deployment, takeoff, hover, and landing flight test (requires honest physics and complexity limits)"),
    ),
    evaluate=evaluate,
)

