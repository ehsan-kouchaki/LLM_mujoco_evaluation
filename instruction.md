# Aerial Manipulator MuJoCo Task

Create a MuJoCo MJCF model at:

`/tmp/output/model.xml`

The model should be an aerial manipulator with the following specifications:

- **Base & Frame**: The main body of the system must be the first body defined within the `<worldbody>` (ensuring it receives body ID 1). It must be named exactly `drone` and have a starting position of exactly `pos="0 0 1.0"`. It must feature a `<freejoint/>` (or equivalent free joint) to enable 6-DOF unconstrained movement in the air. 
The drone must include a central hub geometry named exactly `central_hub` of type `cylinder`, positioned exactly at the local origin (`pos="0 0 0"`). Attached to this body must be four fixed structural arms named exactly `arm_front`, `arm_back`, `arm_left`, and `arm_right`. The arms must extend outward along the +X, -X, +Y, and -Y axes respectively,  all be of the exact same geometry type, share identical size dimensions across all axes, be mirror-symmetric (each arm geometry centre at the same distance from the hub centre), and sit at the exact same vertical level as the central hub (arm geometry centres at local Z = 0). Additionally, the central hub and the four arms must have a height (Z-dimension) that falls within `central_hub_height.tolerance` of `central_hub_height.target`.
- **Landing Gear / Ground Contact Limit**: To prevent the use of excessively wide or un-tippable bases, the system must be designed such that when the drone drops and settles naturally on the ground, a maximum of **two geometries** from the entire model are in direct contact with the ground plane.

- **Structural Contiguity**: The drone frame must be a single, unbroken physical structure. The frame consists of every geometry of the `drone` body, every geometry of any child body rigidly attached to `drone` (a body without joints), and every geometry of the optional extra jointed body attached to `drone` (everything except `link1`, `link2` and their geometries). In the model's default pose (all joint positions at their default values) each of these geometries must physically intersect or touch (surface gap of at most 1 mm) at least one other frame geometry, and together they must form a continuous chain that traces back to the `central_hub`. Floating, disconnected, or invisible-gap geometries are prohibited.

- **Rotors & Propellers**: You must define exactly four actuator sites representing the physical rotor mounts. 
    - **Naming & Placement**: `thruster_front` (on `arm_front`), `thruster_back` (on `arm_back`), `thruster_left` (on `arm_left`), and `thruster_right` (on `arm_right`). All four sites must be defined directly in the `drone` body (not inside a child body); their positions are expressed in the `drone` body frame.
    - **Positioning Constraints**: Each site must sit precisely on the axis line of its corresponding arm in the horizontal plane (zero offset along the perpendicular horizontal axis, on the arm's side of the hub). Its horizontal distance from the hub centre cannot exceed the arm's reach, measured from the hub centre to the outer tip of the arm geometry (including a capsule's hemispherical cap). Its vertical placement (Z-axis) must be at or above the arm's centreline Z-level and no more than `rotor_sites_height_diff_max` above it.
    - **Site Orientation**: Each thruster site's local +Z axis (the thrust direction) must coincide with the drone body's +Z axis. Do not rotate the thruster sites; any tilt beyond numerical round-off is rejected.
    - **Propeller Clearance**: The sites must be spaced far enough apart to physically accommodate propellers with a diameter equal to `propellers_diameter` without any overlap. Do not include explicit propeller geometries in the XML.
    - **Actuators**: You must attach exactly four `<motor>` actuators to these sites to generate vertical thrust and yaw torque. Their gear vectors must be configured exactly as follows:
        1. Front motor at `thruster_front` with `gear="0 0 1 0 0 0.1"`
        2. Back motor at `thruster_back` with `gear="0 0 1 0 0 0.1"`
        3. Left motor at `thruster_left` with `gear="0 0 1 0 0 -0.1"`
        4. Right motor at `thruster_right` with `gear="0 0 1 0 0 -0.1"`

- **Manipulator Topology**: The drone must be equipped with a 2-link manipulator arm attached to the `drone` body:
    - **Hierarchy & Naming**: The arm must consist of exactly two moving links connected in series. The first link must be defined as a direct child of the `drone` body and named exactly `link1`. The second link must be defined as a direct child of `link1` and named exactly `link2`. `link1` may have no child body other than `link2`, and `link2` may have no child bodies.
    - **Geometries**: `link1` must contain exactly one geometry, which must be of type `capsule`. The very first geometry defined in `link2` must be of type `capsule` and must be the longest geometry within that body; we call this the primary geometry of `link2`. You may add additional geometries to `link2` (e.g., an end-effector tool), but only as geometries attached directly to `link2`, not as newly defined bodies.
    - **Link Lengths**: The geometry of `link1` and the primary geometry of `link2` must each have a length of at least `manipulator.link_length_min`, where a capsule's length means the distance between its two `fromto` endpoints (the cylindrical section, excluding the hemispherical caps). The primary geometry of `link2` must be strictly longer than the geometry of `link1`. Specifically, the length of `link2`'s primary capsule minus the length of `link1`'s capsule must be greater than or equal to `manipulator.links_length_diff_min`.
    - **Link Connection**: The two links must be connected sequentially in series. Specifically, the `pos` attribute of `link2` must coincide strictly with the far `fromto` endpoint (distal end of the cylindrical section, not the outer surface of the cap) of `link1`'s capsule-not its base attachment point-and the `fromto` attribute of the primary geometry of `link2` must start from `0 0 0` (e.g., `fromto="0 0 0 x y z"`).
    - **Manipulator Mounting**: The structural base of the manipulator must be defined by a specific geometry of type `capsule` named `arm_base` located anywhere within the `drone` body. The starting origin of `link1` must physically intersect (lie inside or exactly on the surface of) this `arm_base` capsule. The structural attachment point (the `pos` of `link1`, expressed in the `drone` body frame) must be lower than the centre of the central hub (the drone body origin). The vertical distance from the centre of `central_hub` (not its bottom face) down to this point must fall within the range specified by `manipulator.base_pos_to_hub_range` (inclusive).
    - **Mass Requirements**: The total calculated mass of `link1` and the total calculated mass of `link2` must each independently be greater than or equal to the minimum mass specified by `manipulator.links_mass_min`.

- **Joints & Actuators**: The manipulator must have a total of exactly three hinge joints, with all joints having the local position attribute `pos="0 0 0"`. Each joint must be individually actuated by a `<motor>` element:
    1. **Vertical Shoulder Joint**: Named `arm_joint0`, rotating around the vertical (Z) axis with range matching exactly `joints_range.shoulder_vertical`. Actuated by motor `joint0_motor` with `gear="1.0"` and `ctrlrange="-1 1"`.
    2. **Horizontal Shoulder Joint**: Named `arm_joint1`, rotating around a horizontal axis with range matching exactly `joints_range.shoulder_horizontal`. Actuated by motor `joint1_motor` with `gear="1.0"` and `ctrlrange="-1 1"`.
    3. **Horizontal Elbow Joint**: Named `arm_joint2`, rotating around the EXACT SAME horizontal axis vector (identical direction and sign) as `arm_joint1`, with range matching exactly `joints_range.elbow`. Actuated by motor `joint2_motor` with `gear="1.0"` and `ctrlrange="-1 1"`.
    **Anti-Roll Requirement**: `arm_joint1` and `arm_joint2` must act as bending hinges, not twisting roll mechanisms. Their rotation axes must be strictly perpendicular (orthogonal) to the longitudinal length of `link1` and the primary geometry of `link2` respectively.
    **Joint Dynamics Limits**: For all three manipulator joints, the `damping` and `frictionloss` attribute must not exceed the values defined by `joints_range.damping_max` and `joints_range.frictionloss_max` respectively.

- **Sensors**: The drone must be equipped with exactly three sensors mounted on a single `<site>` element:
    - **Sensor Composition**: Exactly one accelerometer (`<accelerometer>`), one gyroscope (`<gyro>`), and one global framepos sensor (`<framepos>`) with attribute `objtype="site"`. No other sensors are permitted.
    - **Site Association**: All three sensors must reference the exact same `<site>` element attached to the `drone` body.
    - **Site Geometry & Height**: The `<site>` element must have a geometry type of `box`, and its total vertical height ($2 \times \text{size}_z$) must not exceed `sensor_site_height_max`.
    - **Site Placement**: The site must be centered horizontally on the `central_hub` ($X = 0, Y = 0$) and mounted directly on top of the `central_hub` cylinder with zero clearance.

- **Mass Constraints**: The mass of the drone must strictly adhere to the following requirements:
    - **Total System Mass**: The total combined mass of the entire system must be equal to `mass.target`, within a permissible margin of error of `+/- mass.tolerance`.
    - **Geometry Mass Distribution**: Every `<geom>` element placed in the `<worldbody>` tree must explicitly include the `mass` attribute, with one strict exception: geometries of `type="plane"` are exempt and should not have a mass. For all other geometries, the value of the mass attribute must fall within the range defined by `mass.geom_range`. Geometry templates inside `<default>` are not geometries and need no `mass`.

- **Maximum System Dimensions**: Evaluated while the drone is fully settled on the ground plane in its default "home" configuration (all manipulator joint angles at 0 degrees). The global bounding box of the entire drone system (excluding the ground plane) is calculated. The total length (X-axis), width (Y-axis), and height (Z-axis) of this bounding box must not exceed the maximum limits specified in `max_dimensions`.

- **Complexity Limits**: To ensure computational efficiency and prevent over-complication of the model, the entire system must strictly adhere to the following limits:
    - **Geometries**: A maximum of 30 `<geom>` elements in total.
    - **Bodies**: You are allowed to add one additional `<body>`. Then in total, maximum of 4 user-defined `<body>` elements (excluding the mandatory `worldbody`).
    - **Joints**: A maximum of 5 `<joint>` elements (including the drone's `freejoint`). If you choose to include an extra joint beyond the drone's root freejoint and the three manipulator joints, its `damping` attribute must not exceed 1.0, and its `frictionloss` attribute must not exceed 0.2.
    - **Motors**: Exactly 7 actuators, all of them `<motor>` elements. No other actuator element types (for example `<general>`, `<position>`, `<velocity>`, `<intvelocity>`, `<damper>`, `<cylinder>`, `<muscle>`, `<adhesion>`) may appear in the `<actuator>` section.
    - **Sites**: Exactly 5 `<site>` elements.
    - **Tendons**: The use of `<tendon>` elements is strictly prohibited.
    - **Flags**: The `<flag>` element must not be used.
    - **Ground Plane & Scenery**: The ground `plane`, placed directly in the `worldbody`, must be the only geometry fixed to the world, and it must be the only `<geom>` placed directly in the `worldbody`. Every other body must belong to the `drone` body tree (the `drone` body and its descendants); no other bodies, static scenery, walls, or scaffolding (welded or free) may be placed in the environment.
    - **Equality Constraints**: The use of `<equality>` tags (e.g., welds, connects) is strictly prohibited.
    - **Physics Solver**: The `<option>` tag must strictly adhere to:
        - `timestep` must be exactly `0.001`.
        - `iterations` must not exceed `150` (default is 100).
        - `tolerance` must be `1e-8` or greater (default is 1e-8; setting it to 0 is prohibited).
        - No other `<option>` attributes may be set, except `gravity` and the default-valued `cone`, `impratio`, `noslip_iterations`, `integrator`, `density`, `viscosity`, and `wind` described below.
    - **Allowed Geometries**: All geometries defined in the model must be strictly of type `capsule`, with exactly two exceptions: the single required ground `plane`, and the `central_hub` (which must be a `cylinder`). Geometries of type box, sphere, ellipsoid, mesh, or any other type are strictly prohibited.
    - **Armature**: If the `armature` attribute is defined for any joint (including the manipulator joints), its value must not exceed `0.0001`.
    - **Engineering Standardization**: To ensure realistic and robust engineering practices, specific parameters must adhere to standardized increments and limits:
    - The use of `quat` or `euler` attributes for rotation on any element is strictly prohibited; all spatial orientations must rely on standard coordinates, axis vectors, or `fromto` mappings.
    - Capsule endpoints must lie on a strict `0.01` m (1 cm) grid: for every `capsule` geometry, however it is defined (`fromto`, or `pos` plus `size` with an axis vector, including attributes inherited from `<default>` classes), both endpoints of its cylindrical section, expressed in the frame of the body that contains it, must have every coordinate in `0.01` m increments (e.g., `0.06` or `-0.15` are valid; `0.062` or `0.105` are prohibited).
    - The `size` attribute of all `capsule` geometries must not contain any value smaller than `0.005`, and every capsule's radius must be at least `0.005`.
    - The `mass` attribute of all geometries must be defined in strict `0.002` increments (e.g., `0.010` or `0.004` are valid; `0.013` or `0.005` are prohibited).

- **Honest Physics & Collisions**: The simulation must rely on standard, realistic physics. 
    - **Gravity**: Global gravity must remain Earth-like (approximately `[0, 0, -9.81]`).
    - **Gravity Compensation**: The use of `gravcomp` (gravity compensation) on any body is strictly prohibited.
    - **Inertia & Mass**: Explicit `<inertial>` tags are strictly prohibited; body inertia and center of mass must be derived naturally from geometry definitions. The `<compiler>` attributes `settotalmass`, `boundmass`, `boundinertia`, `balanceinertia`, `inertiagrouprange`, `inertiafromgeom`, `fusestatic`, and `alignfree` must not be used.
    - **Collisions & Visual Geoms**: All global contact flags must be enabled. You are strictly forbidden from disabling collisions for any geometry. Do not alter `contype` or `conaffinity` (they must remain fully-collidable). Consequently, purely "visual" geometries are not allowed.
    - **No Exclusions, Ghosting, or Explicit Pairs**: The use of `<exclude>` tags to bypass collisions is strictly prohibited. You must not use `gap` or `margin` attributes on geometries to bypass constraint forces (no ghost geometries). Additionally, the use of explicit contact pairs (`<pair>` tags inside a `<contact>` block) is strictly forbidden; all collisions must rely on default spatial intersections.
    - **Naming Restrictions**: No geometry is allowed to have the word `propeller` (case-insensitive) anywhere in its `name` attribute. 
    - **Joint Attributes**: The drone's root `freejoint` must not contain any `damping`, `armature`, or `frictionloss` attributes (it must represent free, unresisted flight). No joint in the model may define a non-zero `stiffness`.
    - **Geometry Attributes**: Geometry sliding friction must not exceed standard bounds (maximum `1.5`). Torsional and rolling friction components must remain at their near-zero defaults, and `condim` must remain at its default value `3` on every geometry. Every geometry must stay in the default `group` `0`. The use of `priority`, `solimp`, or `solref` tags on geometries is strictly prohibited.
    - **Compilation & Memory Limits**: To prevent compiler overloads and memory bombs, you must not use the `<size>` tag. Furthermore, the use of structural macro tags including `<include>`, `<replicate>`, `<composite>`, and `<flexcomp>` is strictly prohibited; the entire model must be explicitly defined within the single file. Unbounded-cost features are also prohibited and are rejected before compilation: no `<flex>`/`<deformable>`, `<skin>`, `<plugin>`/`<extension>`, `<hfield>`, `<mesh>`, `<texture>`, `<frame>`, `<attach>` or `<model>` elements, no `file` attributes, no DOCTYPE/ENTITY declarations, at most 2000 XML elements, and no attribute value longer than 4096 characters. A model that violates any of these scores 0.
    - **Global Solver & Environment Options**: You must not alter global solver options or environment parameters to artificially stabilize the drone. 
        * Attributes like `cone`, `impratio`, and `noslip_iterations` must remain at their default values.
        * The physics `integrator` must remain default (Euler) to prevent numerical dampening.
        * Do not override global flags (the `enableflags` attribute must not be used).
        * Artificial aerodynamic dampening is strictly forbidden; global `density`, `viscosity`, and `wind` must remain zero.


    
- **Collision Sweep Tests**: The drone will be evaluated for self-collision in its initial airborne state. During this test, imaginary propeller discs are injected at each thruster site. No inter-collision (measured strictly via MuJoCo physics contacts) is permitted between the manipulator links (`link1` and `link2`) and any other component of the drone (drone body, arms, or the imaginary propellers) under the following 6 kinematic configurations. **Note: Collisions strictly between `link1` and `link2` are expected at the elbow joint and are therefore ignored.**
    1. `arm_joint0` sweeps through its full limit range, while `arm_joint1` is fixed at 0, and `arm_joint2` is fixed at -30 degrees.
    2. `arm_joint0` sweeps through its full limit range, while `arm_joint1` is fixed at its 90 degrees, and `arm_joint2` is fixed at 0.
    3. `arm_joint0` sweeps through the range of [50, 310] degrees, while `arm_joint1` is fixed at -90 degrees, and `arm_joint2` is fixed at its 90 degrees.
    4. `arm_joint0` sweeps through the range of [-130, 130] degrees, while `arm_joint1` is fixed at 90 degrees, and `arm_joint2` is fixed at 90 degrees.
    5. `arm_joint0` sweeps through the range of [10, 350] degrees, while `arm_joint1` is fixed at -90 degrees, and `arm_joint2` is fixed at -10 degrees.
    6. `arm_joint0` sweeps through the range of [10, 350] degrees, while `arm_joint1` is fixed at -45 degrees, and `arm_joint2` is fixed at -30 degrees.
The range attribute for `arm_joint0` must exactly equal the minimum and maximum values provided in the `joints_range` section of the specification. Do not use smaller subsets of this range.

- **End Effector Clearance**: When the drone is allowed to settle naturally on the ground plane for 5.0 seconds with all manipulator joints left unforced and free (no position/velocity overrides or applied torques), the end effector-defined as the outer tip of `link2`'s primary capsule at its far end from the elbow, including the hemispherical cap (far `fromto` endpoint plus the capsule radius along the capsule axis)-must maintain strict vertical clearances:
    - **To Central Hub**: The vertical distance between the end effector and the mathematical center of the `central_hub` geometry (measuring to its center origin, *not* its bottom face) must be at least `ee_clearance_min.to_central_hub`.
    - **To Ground**: The vertical distance between the end effector and the ground surface must be at least `ee_clearance_min.to_ground`.

- **Push Stability**: Evaluated while the drone is resting on the ground plane with all actuators disabled (`mjDSBL_ACTUATION`). During this test, all manipulator joints are unactuated and not held; they move freely under passive physics dynamics. A force specified by `push_test.force_n` is applied in the positive X direction (+X) directly at the center of the `central_hub` geometry for `push_test.duration_sec`. Throughout the push and subsequent observation period, the maximum angular tilt relative to the drone's baseline settled orientation must fall within the limits defined by `push_test.tilt_range_deg` ([tilt_min, tilt_max], inclusive). Furthermore, at the end of the test, the drone's absolute XY plane displacement must not exceed `push_test.displacement_max`, and its yaw rotation must not deviate from its baseline by more than `push_test.yaw_change_max` degrees.

- **Flight Tests**: The drone's flight-worthiness is evaluated across two dynamic flight missions flown by a fixed, test-supplied cascaded flight controller whose gains are in `flight_test.controller`. It uses ground-truth drone state: an XY position PD loop (holding the take-off XY) produces desired roll/pitch angles (clipped to `tilt_cmd_max_deg`); an altitude PD loop with tilt-compensated gravity feed-forward ($m_{total} \cdot (g + a_z)/\cos\theta$) gives the collective thrust; roll/pitch/yaw PD loops on attitude error and body angular rate give body torques; and a 4x4 mixer, built from your thruster site positions in the `drone` frame and the declared yaw-torque coefficients (+0.1 front/back, -0.1 left/right), turns [collective, roll torque, pitch torque, yaw torque] into the four rotor commands, clipped to `thrust_range`. The controller has no integral term, so a center of mass sitting horizontally off the thrust centroid in a deployed arm pose has to be trimmed by a steady attitude/position error, which shows up as XY drift. Each mission follows this sequence:
    - **Settle Phase**: The drone rests on the ground for `flight_test.settle_time` seconds with no actuation to establish its baseline initial coordinates.
    - **Arm Deployment Phase**: Manipulator joints are driven to target setpoints using a PID controller (parameters in `flight_test.pid_arm`). Depending on the test, this phase executes one or two sequential stages:
        - **Test 1 Targets (Single Stage)**:
            - *Stage 1*: `arm_joint0`, `arm_joint1`, and `arm_joint2` target `flight_test.test_1.joint0_target`, `flight_test.test_1.joint1_target`, and `flight_test.test_1.joint2_target`, respectively. (Note: Test 1 runs for `flight_test.arm_target_duration_1` seconds and skips Stage 2).
        - **Test 2 Targets (Two Stages)**:
            - *Stage 1*: `arm_joint1` targets `flight_test.test_2.joint1_target_1` for `flight_test.arm_target_duration_1` seconds.
            - *Stage 2*: `arm_joint0`, `arm_joint1`, and `arm_joint2` target `flight_test.test_2.joint0_target_2`, `flight_test.test_2.joint1_target_2`, and `flight_test.test_2.joint2_target_2`, respectively, for `flight_test.arm_target_duration_2` seconds.
        - **Stage Verification**: At the end of each deployment stage, all actively controlled manipulator joint positions must reach their targeted setpoints.
    - **Takeoff & Hover Phase**: The drone ascends to a target relative altitude displacement of `flight_test.target_z_displacement` meters and hovers for `flight_test.hover_time` seconds while holding the deployed arm pose. Halfway through the hover, the commanded heading steps by `controller.yaw_step_deg`.
    - **Landing Phase**: The target altitude is linearly interpolated back to ground level over `flight_test.landing_time` seconds, after which all motor thrust is cut.
    - **Success Criteria**: A mission is successful if:
        1. Manipulator joints successfully reach designated target positions at each deployment stage.
        2. Relative $Z$ altitude reaches the target displacement during the hover phase.
        3. At the end of hover, the heading is within `controller.yaw_tolerance_deg` of the commanded heading, and no rotor command hit a `thrust_range` limit during the second half of the hover.
        4. Body tilt never exceeds `controller.max_tilt_deg` during hover and landing.
        5. Final landed resting $Z$ position returns to initial baseline height.
        6. Final landed $(X, Y)$ position differs from initial position $(X_0, Y_0)$ not more than `flight_test.max_xy_drift.target` plus `flight_test.max_xy_drift.tolerance`.

- **Validity of Physical Tests**: The bounding-box, collision-sweep, end-effector-clearance, push, and flight tests describe physical behaviour, so they only count for a model that satisfies every Complexity Limits and Honest Physics rule above; a model that breaks one of those rules fails these five tests as well.

`/data/` contains `spec.json` (every named parameter referenced above, such as `mass.target` or `flight_test.controller`) and `evaluation_protocol.json` (the test procedures and their fixed simulation constants).