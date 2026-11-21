#!/usr/bin/env bash
set -euo pipefail

SPEC_PATH="/data/spec.json"
if [[ ! -f "$SPEC_PATH" ]]; then
  SPEC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)/spec.json"
fi

read -r SV_MIN SV_MAX SH_MIN SH_MAX EL_MIN EL_MAX <<< "$(python - "$SPEC_PATH" <<'PY'
import json
import sys

spec = json.load(open(sys.argv[1]))
jr = spec["joints_range"]

print(
    jr["shoulder_vertical"][0], jr["shoulder_vertical"][1],
    jr["shoulder_horizontal"][0], jr["shoulder_horizontal"][1],
    jr["elbow"][0], jr["elbow"][1],
)
PY
)"

for v in "$SV_MIN" "$SV_MAX" "$SH_MIN" "$SH_MAX" "$EL_MIN" "$EL_MAX"; do
  [[ -n "$v" ]] || { echo "missing parsed spec value" >&2; exit 1; }
done

cat > /tmp/output/model.xml <<XML
<mujoco model="quadrotor">
    <compiler angle="degree" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.001"/>
    <visual>
        <global offwidth="1280" offheight="720"/>
    </visual>    
    <asset>
        <material name="body_mat" rgba="0.2 0.2 0.2 1"/>
        <material name="arm_mat" rgba="0.8 0.4 0.1 1"/>
        <material name="compensation" rgba="0.1 0.8 0.1 0.6"/>
        <material name="rotor_front" rgba="0.8 0.1 0.1 0.6"/> 
        <material name="rotor_back" rgba="0.8 0.1 0.1 0.6"/>
        <material name="rotor_sides" rgba="0.1 0.1 0.8 0.6"/>
    </asset>

    <worldbody>
        <!-- Ground Plane -->
        <light pos="0 0 5" dir="0 0 -1" directional="true"/>
        <geom type="plane" size="30 30 0.1" rgba="0.8 0.9 0.8 1"/>

        <!-- Drone Body -->
        <body name="drone" pos="0 0 1.0">
            <freejoint/>
            
            <!-- IMU Site -->
            <site name="imu_site" pos="0 0 0.03" type="box" size="0.02 0.02 0.01" rgba="0.9 0.9 0.1 1"/>

            <!-- Central Hub -->
            <geom type="cylinder" name="central_hub" size="0.04 0.02" material="body_mat" mass="0.1"/>

            <!-- Compensation Geoms -->
            <geom type="capsule" fromto="0.3 0   0     0.30   0  -0.01" size="0.01" material="compensation" mass="0.01"/>
            <geom type="capsule" fromto="0   0  -0.3 -0.05  0  -0.3" size="0.005" material="body_mat" mass="0.22"/>

            <!-- Arms -->
            <geom type="capsule" name="arm_front" fromto="0 0 0  0.32  0    0" size="0.015" material="body_mat" mass="0.1"/>
            <geom type="capsule" name="arm_back"  fromto="0 0 0 -0.32  0    0" size="0.015" material="body_mat" mass="0.1"/>
            <geom type="capsule" name="arm_left"  fromto="0 0 0  0     0.32 0" size="0.015" material="body_mat" mass="0.1"/>
            <geom type="capsule" name="arm_right" fromto="0 0 0  0    -0.32 0" size="0.015" material="body_mat" mass="0.1"/>

            <!-- Legs -->
            <geom type="capsule" fromto=" 0     0.31  0     0     0.31 -0.34" size="0.015" material="body_mat" mass="0.1"/>
            <geom type="capsule" fromto="-0.31  0.31 -0.35  0.31  0.31 -0.35" size="0.015" material="body_mat" mass="0.2"/>
            <geom type="capsule" fromto=" 0    -0.31  0     0    -0.31 -0.34" size="0.015" material="body_mat" mass="0.1"/>
            <geom type="capsule" fromto="-0.31 -0.31 -0.35  0.31 -0.31 -0.35" size="0.015" material="body_mat" mass="0.2"/>

            <!-- Links Holder-->
            <geom type="capsule" fromto="0   -0.20  -0.25  0      0.20  -0.25" size="0.005" material="body_mat" mass="0.006" />
            <geom type="capsule" fromto="0   -0.20  -0.25  0     -0.20  -0.30" size="0.005" material="body_mat" mass="0.006"/>
            <geom type="capsule" fromto="0    0.20  -0.25  0      0.20  -0.30" size="0.005" material="body_mat" mass="0.006"/>
            <geom type="capsule" fromto="0   -0.20  -0.30  0     -0.32  -0.30" size="0.005" material="body_mat" mass="0.006"/>
            <geom type="capsule" fromto="0    0.20  -0.30  0      0.32  -0.30" size="0.005" material="body_mat" mass="0.006"/>
            <geom type="capsule" fromto="0    0     -0.25  0      0     -0.30" size="0.005" material="body_mat" mass="0.006"/>
            <geom type="capsule" fromto="0    0     -0.28  0.1    0     -0.28" size="0.005" material="body_mat" mass="0.05"/>
            <geom type="capsule" fromto="0.1  0     -0.28  0.1    0     -0.25" size="0.005" material="body_mat" mass="0.01"/>
            <geom type="capsule" fromto="0.1  0     -0.25  0.3    0     -0.25" size="0.005" material="body_mat" mass="0.01"/>


            <!-- Thrust Sites (Rotors) -->
            <site name="thruster_front" pos=" 0.30  0     0.02" type="cylinder" size="0.04 0.01" material="rotor_front"/>
            <site name="thruster_back"  pos="-0.30  0     0.02" type="cylinder" size="0.04 0.01" material="rotor_back"/>
            <site name="thruster_left"  pos=" 0     0.30  0.02" type="cylinder" size="0.04 0.01" material="rotor_sides"/>
            <site name="thruster_right" pos=" 0    -0.30  0.02" type="cylinder" size="0.04 0.01" material="rotor_sides"/>

            <!-- Manipulator Base -->
            <geom type="capsule" name="arm_base" fromto="-0.30 0 -0.02  -0.30 0 -0.04" size="0.01" material="compensation" mass="0.016"/>

            <!--  MANIPULATOR -->
            <body name="link1" pos="-0.30 0 -0.04">
                <!-- Vertical Shoulder Joint (Z-axis) -->
                <joint name="arm_joint0" type="hinge" axis="0 0 1" pos="0 0 0" range="$SV_MIN $SV_MAX" damping="0.05"/>
                <!-- Horizontal Shoulder Joint (Y-axis) -->
                <joint name="arm_joint1" type="hinge" axis="0 1 0" pos="0 0 0" range="$SH_MIN $SH_MAX" damping="0.05" frictionloss="0.2"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.01" material="arm_mat" mass="0.01"/>

                <body name="link2" pos="0 0 -0.20">
                    <!-- Horizontal Elbow Joint (Y-axis) -->
                    <joint name="arm_joint2" type="hinge" axis="0 1 0" pos="0 0 0" range="$EL_MIN $EL_MAX" damping="0.05" frictionloss="0.2"/>
                    <geom type="capsule" fromto="0     0    0 0.40  0     0" size="0.005" material="arm_mat" mass="0.02" />
                    <geom type="capsule" fromto="0.40 -0.03 0 0.40  0.03  0" size="0.005" material="arm_mat" mass="0.002"/>
                    <geom type="capsule" fromto="0.40 -0.03 0 0.40 -0.03 -0.04" size="0.005" material="arm_mat" mass="0.002"/>
                    <geom type="capsule" fromto="0.40  0.03 0 0.40  0.03 -0.04" size="0.005" material="arm_mat" mass="0.002"/>
                </body>
            </body>           

            <!-- Compensation Body -->
            <body name="counterweight_system" pos="0.031 0 -0.28">
                <joint name="cw_slider" type="slide" axis="1 0 0" range="-0.031 0.0" damping="1.0" frictionloss="0.2"/>
                <geom type="capsule" fromto="0 -0.05 0 0  0.05 0" size="0.02 " material="compensation" mass="0.100"/>
            </body>           
        </body>
    </worldbody>

    
    <!-- ================= Actuators ================= -->
    <actuator>
        <!--Rotor Joints-->
        <motor name="thrust_front" site="thruster_front" gear="0 0 1 0 0 0.1"/>
        <motor name="thrust_back"  site="thruster_back"  gear="0 0 1 0 0 0.1"/>
        <motor name="thrust_left"  site="thruster_left"  gear="0 0 1 0 0 -0.1"/>
        <motor name="thrust_right" site="thruster_right" gear="0 0 1 0 0 -0.1"/>

        <!-- Manipulator Joints-->
        <motor name="joint0_motor" joint="arm_joint0" gear="1.0" ctrlrange="-1 1"/>
        <motor name="joint1_motor" joint="arm_joint1" gear="1.0" ctrlrange="-1 1"/>
        <motor name="joint2_motor" joint="arm_joint2" gear="1.0" ctrlrange="-1 1"/>
    </actuator>

    <!-- ================= Sensors ================= -->
    <sensor>
        <!-- IMU (Mounted on top of drone) -->
        <accelerometer name="imu_accel" site="imu_site"/>
        <gyro name="imu_gyro" site="imu_site"/>
        
        <!-- Drone Global Positioning -->
        <framepos name="drone_pos" objtype="site" objname="imu_site"/>
    </sensor>
   
</mujoco>
XML