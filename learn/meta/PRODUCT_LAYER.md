# Product layer for one boat (track C, 2026-10-06)

中文摘要：在不改物理文件的前提下，新建 `app/` 包，补上 LOTUSim 有而我们没有的、对一条船有用的产品层：
Gym 风格的环境接口（按船名和版本后缀选船和物理）、JSON 场景文件、能跑任意策略的运行器、
每次运行一份可逐位复现的记录、带种子的传感器模型、LOTUSim 兼容消息形状的 websocket / UDP / HTTP 接口、
可选的 ROS 2 桥（写好但不要求本机运行），以及网页实时或回放演示和 Blender 导出。
验收按 `learn/meta/ACCEPTANCE_2026-10-06.md` 的 P 节，具体检查在第 1 节事先写定，结果在第 6 节。

2026-10-06 代码复查后的修正（第 7 节逐条列出）：静态文件的路径穿越（反斜杠、编码形式）已堵住；HTTP 会话改为每个客户端一个、各自加锁、闲置过期；
非有限数的指令一律拒绝且不推进仿真，策略中途出错时保存已跑部分的记录并写明原因；不带种子的 reset() 每回合换一个海况（可由第一次的种子复现）；
场景文件按 LOTUSim 的写法读取航点（waypoints / line / circle，块存在即默认启用），裸的 [纬度, 经度] 不再被当成米；
加载时检查所有数值、传感器参数和策略配置，超出 R1 验证范围时给出警告并写入记录；内置策略可选“用传感器”运行（简单估计器）；
记录指纹覆盖物理版本戳、船型、种子和海浪，代码改动后回放会先说明“物理代码已改”；Blender 可以直接导出并渲染短视频；
websocket 协议错误按 RFC 关闭；实时运行数量和内存有上限；位姿消息与 LOTUSim 完全同形。新增“稳健性”测试组（20 秒以内，纳入快速档）。

2026-10-06 物理线第二步（P1b）之后的产品层跟进（第 8 节）：
- **从记录的任意一行接着跑。** 仿真船每一步除了那 14 个状态量，还依赖一些"看不见的状态"：物理线新加的入水状态场（开关 wagner_entry 的"隐藏状态"，船自己用 hidden_state() 交出来），以及旧版就有、一直没记的喷泵指令延迟队列、喷泵吸水程度、砰击计数等。现在每次运行的记录在每一步都存下这些（格式升为第 2 版），从任何一行恢复后接着喂原来的指令，后面每一步都逐位相同；用同一个脚本策略接着跑，整份记录逐位相同。隐藏状态按原样保存，以后物理线再加新的隐藏状态也会自动带上；另有一个"清点"检查，逐个属性比较恢复后的船和原来的船，漏记的东西会被点名。用一条打开 wagner_entry 的测试船验证过，并做了反向对照（不恢复隐藏状态或延迟队列，结果就不同）。
- **查看器按导出的船体线画船。** 网页查看器（demo.html、proto_web.html）和 Blender 渲染原来用旧版公式从"收窄比例""艏部上翘"两个字段重画船体，对 planing_b（船尾收窄、龙骨上翘）画错了（舭线差约 10 厘米、龙骨差约 7 厘米）。现在都用几何模块导出的舭线和龙骨线；在浏览器里核对了 scarab195、planing_b@as1、probe_hullB 三条船，页面上的数和几何模块逐位一致。
- **缓存按物理版本的标签区分。** 产品层自己的简化模型、转向表、服务器的查看器数据都按 physics.cache_tag 命名，物理代码一改就用新文件，不改写旧文件。
- **船型列表写明哪些能跑。** planing_b 旧版不能跑（列表写出原因），只能用 planing_b@as1；直接用旧版名字会被拒绝并提示正确写法。
- **记录刷新。** 四个演示记录按当前 as1 组合重跑；两个旧版记录与上一版逐位相同（只多了新存的几列）。
- 测试：快速档、完整档全部通过（第 8.4 节）。

同日复查后的修正（第 8.6 节，九条意见全部处理）：
- **"清点"检查补上了盲区。** 清点是把恢复后的船和原来的船逐个属性比较，确认没有漏记的状态。原来它跳过所有"可以像函数一样调用的对象"，而喷泵和舵的指令延迟队列恰好是这种对象，所以清点其实没看到它们。现在清点会进入这类对象，也会检查函数里保存的数值；另加一个自检：把延迟队列改动一个值，清点必须点名指出。
- **物理代码的指纹漏了一个文件。** 记录和缓存靠"指纹"（物理代码文件内容的哈希）判断代码是否改过。新的波浪压力计算模块（`sim/as_kernels.py`）随 fk_exact 开关进入了 as1 的默认组合，却不在指纹清单里：改了它，旧记录仍会被当成同一份代码接着跑，缓存也会被误用。清单归物理线改；产品层新增一项检查：在全新进程里打开所有开关建船并走几步，凡是实际运行到的仓库文件都必须在清单里。这项检查一开始正确地报出了这个文件缺失；本轮也实际发生过一次：这个文件在两次测试之间被改动，船的缓存标签却没有变。物理线随后把它加进了清单，检查现已通过。
- **长时间运行的服务器不再悄悄用旧代码。** 服务器启动时一次性载入全部物理代码并记下指纹；之后若磁盘上的代码被改动，服务器拒绝新的实时运行并提示重启，查看器标题也会写明。
- **船首画得更准。** 导出的舭线和龙骨线原来是均匀取 201 个点，船首最后一厘米处舭线收得很陡，画出来最多偏差四五厘米。现在按曲线弯曲程度自动加密取点，全船（含船首）与几何模块的偏差都在 0.25 毫米以内；网页船体网格直接用这些点，水面效果用的小表也按同法取点（偏差不到 1 毫米）。Blender 渲染在缺少这些线时退回到船的站位数组，不再用旧公式。
- **状态编码更严格。** 零维数组、列优先存放的数组、非数值的二进制位都能原样保存；无法精确保存的对象数组在保存时就报错，而不是等到恢复时才出错。
- **简化模型缓存不再跨代码版本借用。** 原来物理代码改动后，若三秒静水试航的结果相同，就沿用旧版本辨识出的简化模型；但这段试航到不了高速、波浪和转向，所以现在代码一改就重新辨识（每条船约一分钟）。另外如实记下：本轮早期一次实验改写过一份旧缓存文件，违反"旧缓存不得改写"的要求，已改名保留。
- **测试补齐。** 恢复时安全模式（大横摇时自动减速的保护）的状态是否还原；四个 P1b 开关全开时能否从任意一行接着跑；带 wagner_entry 的测试船随默认开关组合自动更新。
- 两份 as1 演示记录按最新代码重跑（物理线补上清单后又重跑了一次，18:22），从中途接着跑与原记录逐位相同；旧版演示的查看器文件也按新画法重新导出（记录本身不变，重跑仍逐位相同）。物理线之后若再改代码，需再重跑一次。服务器的“拒绝旧代码”也在实际中触发过一次：物理线改了代码后，正在运行的服务器拒绝了新的 as1 实时运行，重启后恢复。
- 测试：完整档 176 项中 175 项通过（唯一未通过的就是上面那项清单检查，当时清单还没补）；之后的快速档 175 项中 174 项通过，未通过的一项是测试进行中物理线恰好修改清单造成的前后不一致，单独重跑即通过。
- 服务器的查看器缓存文件名带上导出格式版本，换了画法不会再拿旧文件出来；“服务器代码已过期”的提示在每次提供数据时现算。

Scope: one vessel, no multi-vessel, no ROS install, no change to any physics or
controller file (the files owned by the physics track are imported, never
edited). Everything new lives in `app/`, `viewer/demo.html`,
`studies/app_test.py`, `app/scenarios/*.json`.

## 1. Checks written before implementation (ACCEPTANCE P, N, T)

| id | check | pass criterion |
|---|---|---|
| C-P1a | Gym API conformance | `gymnasium.utils.env_checker.check_env(UsvEnv(...))` raises nothing; reset/step signatures and spaces as below |
| C-P1b | hull and physics selection by key | `scarab195` builds `PlaningVesselGZ` (physics `v0`), `scarab195@as1` builds `PlaningVesselAS` (physics `as1`), `physics.verify(plant, hull)` passes for both; `probe_hullB` (the second hull of `studies/principles_hull_swap.py`) and `probe_hullB@as1` build; an unknown key raises `KeyError` |
| C-P1c | same physics as the evaluation (N1) | the app env + hand MPC on scarab195 reproduces `learn/repro/task.Mission` (`plant: lin`) and `studies/final_eval.make_mission` (`plant: gz`) **bit for bit** (states and commands, 40 control steps, `==`) |
| C-P2a | any policy | scripted, hand MPC (`control/mpc.py`), m0 constrained MPPI, a learned row via `studies/final_eval.py` builders, and any callable run through one runner; m0 and learned rows refuse a hull whose actuator scales differ from the model they were trained on (scarab195) |
| C-P2b | demo | the browser viewer shows a replay (exact recorded sea and poses) and a live run streamed from the app server; screenshots saved under `viewer/demo_shots/`; Blender `scene.json` exported in the format `viewer/proto_blender/render.py` reads |
| C-P3 | scenario files | JSON scenario with hull key, physics version, sea, track or waypoints, duration, seeds, sensors, policy; load -> save -> load is identity; LOTUSim key names where LOTUSim has them |
| C-P4a | one record per run | trajectory at every plant step, sea components, commands, sensor readings, scenario, physics stamp, code version |
| C-P4b | deterministic replay | (i) the same scenario + policy in two separate processes gives the same record fingerprint (sha256 over every array); (ii) feeding the recorded commands open loop into a fresh env reproduces every state and sensor reading `==`; (iii) save -> load preserves the fingerprint |
| C-P5 | sensors | IMU, GNSS, heading, speed log, wave probe, wave preview; with noise off they equal the truth (to 1e-9; GNSS geodetic round trip < 1 mm); with noise on: sample sd within the chi-square 99.9 % band, dropout share within the binomial 99.9 % band, latency exactly the configured value on the sample grid; same seed -> same readings; sensor noise does not change the sea or the trajectory |
| C-P6 | external interfaces | websocket and UDP lockstep sessions reproduce the direct env bit for bit (20 steps); the server-sent-events stream of a live run delivers header, frames and end; message shapes are LOTUSim's (`VesselPositionArray` with ENU pose + `geo_point`, `VesselCmdArray` with a JSON `cmd_string`); the ROS 2 bridge module imports without rclpy and documents its use |
| C-P7 | end-to-end on two hulls | scarab195 and probe_hullB each run a waypoint scenario under the hand MPC (and a scripted run on `@as1`), finish without non-finite states, export viewer JSON and Blender scene |
| C-T | test cost | quick tier (everything above but the learned-row smoke) under 10 min wall clock; independent groups in parallel processes sized by free RAM (`studies/parallel_runner.py`, reserve 3 GB); closed-loop MPC only where the controller interface itself is under test (C-P1c, C-P2a, C-P7: short missions) |
| C-R | robustness (added after the 2026-10-06 code review; the review's probes made into checks) | group `robust`, plant-only (one 10-step hand run with sensors), under 30 s, in the quick tier: static paths with backslash / encoded / drive / `..` forms 404 (GET and HEAD, raw sockets); 4 concurrent HTTP clients x 5 steps each get k = 1..5 and their own direct-env poses; NaN / inf / non-numbers refused on the env, `cmd_string`, websocket, UDP and HTTP with the episode untouched; a policy raising or returning NaN mid-run gives a partial record with reason and traceback, `close()` runs, a live run saves it; LOTUSim `waypoint_interface` (waypoints `[{lat, lng}]`, circle, line; enabled by default) loads, a bare `[lat, lon]` path is refused; a resent UDP step (same `seq`) does not step twice, gaps are refused; dropout 1 is valid, > 1 refused; rate 0 / < 0, negative latency, duplicate preview names, `n_freq` 0, seeds 3.7 / -1, non-numeric values, unknown policy types and bad schedules refused at load with the key named; Hs 5 m loads with an R1 warning recorded in the header; Blender export windows checked, export without an output is an error; JSON strings never altered; websocket protocol errors close with 1007 / 1002 / 1009; live-run limits answer 429; the fingerprint covers stamp, hull, seeds and sea and `code_changes` names what changed; `reset()` without a seed gives a new, reproducible sea; with `state_source: sensors` the sensor errors reach the loop |

## 2. Interfaces

### 2.1 Hull keys (`app/hulls.py`)

`resolve(key)` -> hull. `key` is `NAME` (physics v0) or `NAME@VERSION`
(`sim/physics.py` grammar), where NAME is in `hydro/hulls.REGISTRY` or in the
app's extra table (`probe_hullB`, built by
`studies/principles_hull_swap.make_hull("hullB")`), or a dict of
`PlaningHull` arguments with a `name` (inline hull). Versions go through
`physics.variant`, so caches keyed by name keep versions apart (N3).
`listing()` (`python -m app hulls`, `GET /api/hulls`) lists every registered
and extra hull as `NAME` and, for a planing hull, `NAME@as1`, each with
`runnable` decided by the physics selector itself (`physics.plant_class` /
`gz_class`): `planing_b` is listed as not runnable on v0 with v0's reason
and "run it as planing_b@as1"; `UsvEnv` refuses such a key with the same
message (section 8).

### 2.2 Environment (`app/env.py`)

```python
env = app.UsvEnv("app/scenarios/demo_scarab195.json", seed=0)  # or a dict / Scenario
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)
```

* control step = `sub` plant steps (`sub = round(dt_ctrl / dt)` from
  `Hull.scales()`; 12 x 0.02 s for scarab195, as `Mission`).
* `action`: array `[throttle, nozzle]` (fractions of `t_max` and of full
  nozzle lock; throttle may go down to `t_min / t_max` = -1/3, reverse) in
  `action_mode="direct"`, or `[throttle, heading]` (heading ENU rad, the
  project's heading autopilot `sim/env.Episode._steer` sets the nozzle) in
  `action_mode="heading"`. A dict action with physical units is also
  accepted (`thrust` N, `nozzle_rad`, `heading`), and is what the
  model-based policies send, so no unit round trip touches the bits.
* seeds (Gymnasium's rule): `reset(seed=s)` builds the episode from `s`;
  the first `reset()` without a seed uses the scenario's `usv.seed`; every
  later `reset()` without a seed draws the episode seed from `np_random`, so
  an RL loop sees a new sea each episode and the sequence is reproducible
  from the first seed. The episode seed is `info["seed"]` and
  `header["seed"]`.
* a malformed or non-finite action (NaN, inf, a string) raises
  `app.env.ActionError` (a `ValueError`) before anything is stepped; the
  episode goes on.
* `obs`: `Dict` of the sensors' latest delivered readings,
  `[valid, age_s, channels...]` each, plus `truth` (`[t, state14]`) when the
  scenario sets `obs_truth`.
* `reward = -` the operator cost of `learn/repro/task.py` (speed made good
  along the current leg, CG impacts above 1 g, cross-track per 5 L),
  averaged over the control step: exactly `Mission.advance` on a straight
  track.
* `terminated`: capsize (GZ plant), non-finite state, or the last waypoint
  reached; `truncated`: duration reached.
* `info`: `t`, `k`, `seed`, `state`, `leg`, `n_legs`, `cost_parts`,
  `capsized`, `finite`, `reason`, `cmd`, `hull`, `physics` (stamp).

### 2.3 Scenario (`app/scenario.py`, JSON; YAML when PyYAML is installed)

LOTUSim's keys where LOTUSim has them (`name`, `description`,
`reference_position`, `environment.ocean.wave_height / wave_period /
wave_direction / wave_spectrum`, `agents.<name>.model / position / heading /
waypoint_interface`), our own under `usv` and in the agent:

```json
{"name": "demo_scarab195", "reference_position": {"latitude": 1.2606, "longitude": 103.7516, "altitude": 0.0},
 "environment": {"ocean": {"wave_height": 1.0, "wave_period": 5.0, "wave_direction": 90.0,
                           "wave_spectrum": "JONSWAP", "gamma": 3.3, "spreading_s": 8, "n_freq": 24, "n_dir": 5}},
 "agents": {"usv0": {"model": "scarab195@as1", "plant": "gz", "position": {"enu": [0, 0]}, "heading": 0.0,
                     "speed": null, "waypoint_interface": {"path_enu": [[400, 0], [700, 300]], "radius": 25},
                     "sensors": {"gnss": {"sigma_h": 1.0, "dropout": 0.05}},
                     "policy": {"type": "hand", "frame": "heading", "state_source": "truth"}}},
 "usv": {"duration": 60.0, "seed": 0, "safe_mode": false, "obs_truth": false}}
```

`waypoint_interface` is read the way LOTUSim's
`systems/lotusim_mas/src/scenario_parser.cpp` (`appendWaypointFollower`)
and `systems/waypoint_follower` read it:

| key | meaning |
|---|---|
| `enabled` | default **true** when the block is present (LOTUSim's default); `false` gives the straight track |
| `loop` | default false |
| `mode: waypoints` + `waypoints: [{lat, lng}, ...]` | WGS84 degrees, converted on the tangent plane at `reference_position` (a point may also be `{latitude, longitude}` or `{enu: [x, y]}`) |
| `mode: line` + `line: {direction, length}` | one leg from the start; `direction` in radians, ENU, counter-clockwise from east (an SDF angle in LOTUSim), `length` in m |
| `mode: circle` + `circle: {radius}` | LOTUSim's eight points `start + radius (cos k pi/4, sin k pi/4)`, k = 0..7 (a circle centred on the start) |
| `path_enu: [[x, y], ...]` (ours) | metres, ENU |
| `path: [{enu: [x, y]} or {latitude, longitude}, ...]` (ours) | bare `[a, b]` lists are refused here, so a `[lat, lon]` pair is never read as metres |
| `radius` (ours) or LOTUSim's `range_tolerance` | m, > 0, default 20: a leg ends within it of its end point, or when the boat passes the end point along the leg |

There is no `path` key in LOTUSim; an enabled block that names no
destination, an unknown mode or an unknown key is an error. Validation at
load (`ScenarioError` naming the key): every number finite and in range
(`wave_height >= 0`, `wave_period > 0`, `n_freq`, `n_dir` integers >= 1,
`band` 0 < lo < hi, seeds integers >= 0, `duration > 0`, coordinates in
range), sensors (`app/sensors.validate_config`: `rate_hz > 0` or `null` =
every plant step, `latency_s >= 0`, `0 <= dropout <= 1`, every sigma, bias,
walk and scale >= 0, preview distances distinct to the centimetre -- the
channels are `eta_<d>m`), and the policy (`make_policy` with its schedules
dry-run; a callable's target is checked for its form and never imported at
load). Outside ACCEPTANCE R1 (Hs > 1.5 m, Tp outside 3-10 s, initial speed
outside -3..40 kn) the scenario loads with a `ScenarioWarning`, kept in
`Scenario.warnings` and in every record's `header["scenario_warnings"]`.

Conventions: ENU world (x east, y north, z up), FLU body, yaw CCW from east
(LOTUSim's agent `heading`, "relative to East"); `wave_direction` is where
the waves come FROM, degrees clockwise from true north (LOTUSim's "true north
reference"); internally `theta0` (direction of travel, ENU) = -90 deg -
wave_direction, so 90 (from the east) is `theta0 = pi`, the project's head
sea for a boat heading east. Wind and current are refused unless zero (not
modelled in the planing plant).

### 2.4 Policies and runner (`app/policies.py`, `app/runner.py`)

`Policy`: `action_mode`, `reset(env)`, `act(obs, info) -> action`, optional
`summary()`. Built by `make_policy(spec)`:

| spec | what |
|---|---|
| `{"type": "scripted", "throttle": [[t, f], ...], "heading": "track" or [[t, deg], ...]}` | piecewise-linear schedules; heading via the autopilot |
| `{"type": "hand", "guidance": "los" or "hold"}` | `control/mpc.MPPIController` with the hand weights (`studies/sim2real_jet.z_hand`), thrust only, autopilot steering, as the evaluation's `hand` row |
| `{"type": "m0", "steer": "auto"}` | `learn/meta/mpc_constrained.ConstrainedMPPI` with the e = 0 model (scarab195's) |
| `{"type": "learned", "row": "cat_p_prior"}` | a `studies/final_eval.py` net row (`resolve_row`, `build_net`, `LiveDataW`, learners) stepped one control step at a time |
| `{"type": "callable", "target": "pkg.mod:function"}` or a Python callable | any `f(obs, info) -> action` |

`run(scenario, policy, seed=, duration=, rate=, on_step=, raise_on_error=)
-> Record`.

* **What the built-in policies see.** `scripted` and `hand` take
  `state_source`: `"truth"` (default; the plant state and the env's heading
  autopilot -- the evaluation's setting, so the configured sensor errors are
  recorded but do not reach the loop) or `"sensors"`
  (`app/estimator.SensorEstimator`: GNSS fix minus lever arm plus GNSS
  velocity x age, heading sensor plus yaw rate x age, IMU attitude and the
  inverse ZYX kinematics for the rates, speed log for u, GNSS height and
  vertical velocity; actuator states as measured feedback; the policy's own
  copy of the autopilot steers on the estimate and sends the nozzle
  directly). m0 and learned rows run on the truth, as `final_eval` does;
  callables and external clients read the observation. The viewer's HUD
  shows it ("controller sees").
* **Failures.** A policy whose `act` / `observe` raises, or whose action
  is refused (`ActionError`), ends the episode with reason
  `policy_error: <type>: <message>` (`env_error: ...` when the plant itself
  raised); the traceback goes into `header["error"]`, `close()` runs, the
  partial record is returned (`raise_on_error=True` re-raises after that).
  Set-up failures (unknown type, a row refusing a hull) raise at once.
* **Labels.** `policies.label(spec)`: e.g. `hand (heading frame)`,
  `hand (heading frame, sensors)`, `scripted`, `m0`; used in run ids, the
  viewer meta and the HUD.

### 2.5 Record and replay (`app/record.py`)

One `.npz` per run: `header` (JSON: scenario, hull key, physics stamp, plant
class, policy, seeds, sea components `a, k, th, w, phi` and spectrum
parameters, time steps, git commit, wall times), plant-step arrays (`t`,
`s[n,14]`, `a_cg`, `a_bow`, `n_sub`, applied `thrust`, `nozzle`; since
format 2 also `aux` and, for a plant with a hidden state, `hidden`: the
plant's state outside `s` after every step, section 8), control-step
arrays (`t`, action as given, applied command, reward, cost parts, leg, safe
mode; since format 2 also `ap`, `safe_state`: the env's autopilot and safe
mode after every step) and per sensor (`t_meas`, `t_avail`, `valid`,
`values`). `replay(record, start=k)` resumes after control step k and
`plant_at(record, j)` gives a plant at plant row j (section 8).
`fingerprint()` = sha256 over the header's identity (`scenario_sha256`,
physics stamp, `hull_sha1`, the three seeds, the sea components; canonical
JSON) and every array. `replay(record)` rebuilds the env from the recorded
scenario (unchanged, so its sha256 matches) and the episode seed, feeds the
applied commands open loop and returns the new record.
`record.code_changes(record)` compares the record's stamp and hull sha1
with what the current code builds for the same key ("physics code changed
since recording (as1: 0bf993194a71 -> 108c00c5e52d)", "as1's preset
switches changed ..."); `python -m app replay`, `export` and the viewer
(title, `meta.code_changes`) report it.

### 2.6 External interfaces (`app/server.py`, `app/lotus.py`, `app/wsproto.py`)

`python -m app serve --port 8766 --udp 8767` (stdlib only: `http.server`,
an RFC 6455 websocket written in `app/wsproto.py`, UDP sockets):

* static viewer and records (`safe_static`: after unquoting, no
  backslash, colon, percent sign, NUL or `.` / `..` component; the real path
  must lie inside `viewer/`, `app/runs/` or `app/scenarios/` by
  `commonpath`), `GET /api/hulls` (with a `runnable` flag) `| scenarios |
  runs`, `POST /api/run` (live run thread; 429 beyond 4 light runs, 1 m0 /
  learned run, or when free RAM would fall under a 3 GB reserve; learned
  rows run at rate 0), `GET /api/stream/<id>` (server-sent events; a
  finished run's frames are dropped 30 s after its record is saved, and a
  late reader gets a `replay` event pointing at the record), `POST
  /api/snapshot` (saves a PNG from the page),
* `/ws` websocket, `POST /api/session` and the UDP port: lockstep JSON
  sessions `{"op": "reset", "scenario": ...}`, `{"op": "step", "cmds": [{"vessel_name": "usv0", "cmd_string": "{\"throttle\": 0.7, \"heading(deg)\": 0}"}]}`,
  replies `{"op": "state", "poses": <VesselPositionArray>, "usv": {...}, "sensors": ..., "reward": ...}`.
  HTTP and UDP sessions live in `lotus.SessionTable`: HTTP's reset reply
  carries `"session": <id>` that every later message carries; one lock per
  session (concurrent clients never step each other's boat), idle sessions
  dropped after 600 s, at most 32. An optional integer `seq` makes a resent
  message idempotent (same `seq` = the cached reply, nothing stepped; a gap
  is refused). UDP datagrams are served by a pool of worker threads, so one
  client's reset does not block the others. Session `save` never
  overwrites a record. Websocket protocol errors close the connection with
  1002 / 1007 / 1009 (unmasked client frames, bad UTF-8, a message over
  4 MB in all its fragments); nothing HTTP is ever written into the
  upgraded stream. All JSON is strict: non-finite numbers become `null`,
  strings are never altered.

Message shapes follow `LOTUSim/interfaces/lotusim_msgs`: `VesselPositionArray
{header, vessels: [{vessel_name, model_name, pose {position, orientation (FLU)},
geo_point}]}` -- the `poses` object is exactly `{header, vessels}`, so a
strict ROS converter accepts it -- and `VesselCmdArray {cmds: [{vessel_name,
cmd_string}]}`; our extras (velocity, actuators, time, leg, reward) are the
reply's sibling key `usv`. A `cmd_string` value that is not a finite number
is refused. `cmd_string` keys use xdyn's `name(unit)`
style: `jet(T)` [N], `nozzle(rad)`, `nozzle(deg)`, `heading(rad)`,
`heading(deg)`, plus `throttle` and `nozzle` fractions.

`app/ros2_bridge.py`: optional, imports rclpy lazily; publishes
`/lotusim/poses` (lotusim_msgs when installed, else JSON on std_msgs/String),
`sensor_msgs/Imu`, `sensor_msgs/NavSatFix`, `nav_msgs/Odometry`, subscribes
`/lotusim/vessel_cmd_array`; lockstep on a ROS timer with `/clock`. Its
flags are argparse flags (`--rate`, `--wait-for-command`, `--remote`), its
stamps use `lotus.stamp` (nanosecond carry), and it stops its timer once the
episode ends.

### 2.7 Demo (`viewer/demo.html`)

Built on `viewer/proto_web.html` (same exact-sea shaders, hull from the
exported station arrays -- `x_fine`, `yc_fine`, `zk_fine` from
`hydro/hull_geometry`, section 8 --, artistic layers marked): `?data=<viewer json>` replays a
record, `?live=<run id>` follows a live run over server-sent events (and replays
its record when its frames were already dropped); adds the hull key, the
policy label, what the controller sees (true state / sensors), commands,
waypoint buoys floating on the exact surface and a track minimap; a record
made by other physics code says so in the title. `app/export.py` writes the
viewer JSON, the Blender `scene.json` (`viewer/proto_blender/render.py`
input format) and the render command: a still, or `clip=(a, b)` frames
plus an MP4 (`find_blender()`: `$BLENDER`, `PATH`, or Program
Files/Blender Foundation). The export window must satisfy
0 <= t0 < t1 <= t_end.

## 3. Files

| path | what |
|---|---|
| `app/__init__.py` | package, lazy exports (`UsvEnv`, `run`, ...) |
| `app/hulls.py` | hull keys: registry, `probe_hullB` (from `studies/principles_hull_swap.make_hull("hullB")`), inline hulls, `@VERSION` via `sim/physics.variant` |
| `app/geo.py` | WGS84 geodetic <-> ECEF <-> ENU, ZYX attitude, quaternions, bearings, wave-direction conversion |
| `app/scenario.py` | scenario files (JSON; YAML with PyYAML), validation, LOTUSim key names |
| `app/track.py` | straight track / waypoint legs, leg switching, `hold` and `los` guidance |
| `app/sensors.py` | IMU, GNSS, heading, speed log, wave probe, preview; seeded errors, latency, dropout |
| `app/models.py` | reduced-model parameters (lofi's cache rule, never rewrites project caches), the autopilot (`Episode._steer` itself), turn table; process caches keyed by name + `physics.key_sha1` |
| `app/estimator.py` | `SensorEstimator`: the plant-state estimate from the sensors (`state_source: sensors`) |
| `app/env.py` | `UsvEnv` (Gymnasium), safe mode, recording |
| `app/policies.py` | scripted, hand MPC, m0 / learned rows (`ConstrainedPolicy` over `MissionView`), callables |
| `app/runner.py` | `run(scenario, policy, ...) -> Record` |
| `app/record.py` | `Record` (.npz), fingerprints, `replay` (from any control step), `plant_at` (any plant row), `RecordMismatch` |
| `app/plantstate.py` | the plant's state outside `s` (physics hidden state through an exact codec, v0's carried actuator state), restore, completeness audit (section 8) |
| `app/export.py` | viewer JSON, Blender `scene.json`, Blender command line |
| `app/lotus.py` | LOTUSim-shaped messages, `Session` lockstep protocol |
| `app/wsproto.py` | RFC 6455 websocket server side and client (stdlib) |
| `app/server.py` | HTTP + server-sent events + websocket + UDP server, live runs |
| `app/ros2_bridge.py` | optional ROS 2 node (not run here) |
| `app/__main__.py` | `python -m app hulls | run | replay | export | serve` |
| `app/scenarios/*.json` | `demo_scarab195`, `demo_hullB`, `demo_scarab195_as1`, `demo_planing_b_as1` (waypoint courses), `eval_head_seas` (the evaluation's geometry, m0) |
| `viewer/demo.html` | the demo viewer (generated from `viewer/proto_web.html` by anchored replacements; the prototype is unchanged) |
| `studies/app_test.py` | the tests of section 1 |
| `.claude/launch.json` | added config `app` (the app server on 8766, UDP 8767) |

Nothing outside these was edited. The physics track's files (`sim/planing_*`,
`sim/physics.py`, `hydro/*`, `control/mpc.py`, ...) were edited by that
track while this one ran; this layer only imports them. Once, mid-edit,
`sim/planing_anyspeed.py` called a `physics.stamp_for` that `sim/physics.py`
did not have yet, and the `@as1` checks failed for that run; they pass on
the consistent files.

## 4. Demo

```
python -m app serve --port 8766 --udp 8767          (or the launch config "app")
http://127.0.0.1:8766/viewer/demo.html?live=new&scenario=demo_scarab195.json     live run, followed
http://127.0.0.1:8766/viewer/demo.html?run=demo_hullB_hand                      replay of a record
http://127.0.0.1:8765/viewer/demo.html?data=../app/runs/demo_scarab195_hand.viewer.json   any static server
python -m app run demo_hullB.json --out NAME --blender 40:44                    record + viewer JSON + Blender scene
"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup --python viewer/proto_blender/render.py -- --scene app/runs/demo_scarab195_hand.scene.json --engine EEVEE --res 960x540 --samples 16 --out viewer/demo_shots/blender_scarab195_hand_eevee.png
python -m app export app/runs/demo_scarab195_hand.npz --blender app/runs/demo_scarab195_hand.scene.json --t0 40 --t1 41 --fps 30 --video viewer/demo_shots/blender_scarab195_hand_clip.mp4 --render --res 640x360 --samples 8
http://127.0.0.1:8766/viewer/demo.html?live=new&scenario=demo_hullB.json&rate=4&policy={"type":"hand","frame":"heading","state_source":"sensors"}
```

After the review fixes (2026-10-06, server restarted on the new code):
replays `fix_replay_hullB_hand_chase_t40.png`, `fix_replay_hullB_hand_side_t60.png`, `fix_replay_scarab195_hand_chase_t41.png`,
`fix_replay_scarab195_as1_scripted_t30.png`; live runs
`fix_live_scarab195_hand_blenderview_t45.png` (scarab195, hand, heading frame,
true state; its record `app/runs/20261006_034513_demo_scarab195_hand_heading_frame.npz`
has the batch run's fingerprint `a6c8c417bb9bab34`),
`fix_live_hullB_hand_sensors_t60.png` (probe_hullB, hand on the sensors: the
course in 90.2 s against 86.2 s on the true state); a late reader of a
finished live run got the `replay` event and the whole record (4549 frames).
Blender video: `viewer/demo_shots/blender_scarab195_hand_clip.mp4` (1 s, 31
frames, EEVEE 640x360, 8 samples; 466 s wall: 0.9 s render + 13.9 s water /
foam / spray scene update per frame, the bottleneck is render.py's per-frame
CPU water mesh, not the GPU), frames in `blender_scarab195_hand_clip_frames/`.

Screenshots (`viewer/demo_shots/`, saved by the page itself through
`POST /api/snapshot`: the WebGL frame with HUD, minimap and timeline
composited): `replay_scarab195_chase_t41.png`, `replay_hullB_hand_side_t33.png`,
`replay_hullB_hand_chase_t64.png`, `live_hullB_hand_t47.png`,
`live_hullB_hand_blenderview_t70.png`, `live_scarab195_hand_blenderview_t21.png`,
`live_scarab195_hand_ended_t91.png`; Blender (EEVEE) from an app record:
`blender_scarab195_hand_eevee.png` (22 s wall).

A live run streamed at real time from the server thread is bit-identical to
the same scenario run in a test process: record fingerprint `450f95453a24c4c3`
both (`app/runs/20261006_023934_demo_scarab195_hand.npz` vs
`studies/_cache/app/test_runs/demo_scarab195.npz`).

## 5. Findings and limitations

* **Hand MPC on waypoint courses.** `control/mpc.MPPIController` plans thrust
  only and its rollouts hold the nozzle at zero, so after a course change the
  leg frame shows a large heading error, the rollouts predict cross-track
  error growing with speed, and the plan cuts the thrust -- which removes the
  jet's steering authority. Seen on probe_hullB at the second 45 deg turn:
  throttle about 1 %, nozzle hard over, 6 kn, course not finished in 90 s.
  `HandMPCPolicy(frame="heading")` plans in a frame along the boat's own
  heading (thrust for speed and seakeeping, course left to the autopilot);
  with it probe_hullB completes the course in 86 s and scarab195 in 91 s.
  The default stays `frame="leg"`, which is Mission bit for bit; the demos'
  label says `hand (heading frame)`. The real fix is a hand MPC whose
  rollouts include the autopilot (m0 already does,
  `learn/meta/mpc_learned.SteerT`) -- a controller change for the track that
  owns `control/mpc.py`, not done here (open task).
* **m0 and learned rows** plan with scarab195's identified model and actuator
  scales and in the evaluation's straight-track frame: they refuse other
  hulls (`mpc_learned.check_env`) and waypoint tracks. Learned rows run at
  about 2.2 s per control step (GPU), ten times slower than real time: use
  replay, not live view, for them. Carry rows (learning across missions)
  stay in `studies/final_eval.py`.
* **Physics scope.** The planing plant only (all its versions). Displacement
  hulls (`wigley10`, KCS, ...) are refused with a clear error. Wind and
  current are refused unless zero (not modelled). One boat; no multi-vessel.
* **Sensors** are models of the measurement, not of the hardware: the IMU's
  specific force is the plant's mean acceleration over the sample interval
  (exact for an integrating accelerometer), its rates the body rates of the
  plant's Euler-angle rates; GNSS errors are white noise plus a constant bias
  and a random walk (no multipath, no satellite geometry); the wave probe
  and preview read the incident sea, not a radar return. Sampling and latency
  are quantised to the plant step (0.02 s); the actual values are reported.
* **Safe mode** is a copy of `mpc_constrained.SafeMode` (to keep torch out of
  plant-only processes), checked equal on 5000 random steps.
* **Interfaces** are 127.0.0.1 only, no TLS, no authentication beyond the
  same-origin rule; callable policies are refused over HTTP, also when they
  come in an inline scenario's own policy. The websocket has no extensions
  (no compression). A session's env records every step for `save`: a
  session kept open for hours grows by about 10 kB per simulated second
  until it is closed or expires.
* **Hull objects** are resolved once per key and process
  (`app/hulls.resolve`): the plant caches its running attitude and
  righting-arm table per hull object, so every env after the first on a
  key builds in about 10 ms instead of about 3 s; the numbers are the same
  (each cache holds what the same computation returns). The ROS 2 bridge is written against
  the ROS 2 message APIs and LOTUSim's interface definitions but was not run
  (no ROS 2 here); its typed path needs `lotusim_msgs` built in the
  workspace, otherwise it publishes the same JSON on `std_msgs/String`.
* **Viewer.** The hull drawing is the planing hull's stations; console, rails
  and the like are the prototype's (drawn for hulls of 4-12 m, artistic). In
  a hidden browser tab the page does not animate (requestAnimationFrame
  stops); `window.__viewer.setTime / snap / saveShot` still render.
* **Cost.** The env adds about 0.05 ms per plant step without sensors and
  about 0.3 ms with all six (IMU every plant step): 2.2 ms per plant step
  against 1.86 ms for `plant.step` alone (scarab195, best of 3,
  `scratchpad/bench_env.py`). The hand MPC costs 0.031 s per control step,
  so live runs keep real time with a margin of about 4x.

## 6. Results against section 1 (`python -m studies.app_test --tier full`)

After the review fixes: **141 of 141 checks pass, 233 s wall** (full tier;
free RAM was 3.3 GB because another track's job held 4.4 GB, so the pool ran
one process at a time: light groups 200 s, then the torch group 33 s); log
`studies/_cache/app/test_full_fix.log` (`test_results.json` holds the last run).
Quick tier 140 / 140 in 97 s on the final code
(`test_quick_final.log`, 8.9 GB free, parallel); the robust group alone takes 9-24 s depending on the
machine's load.

| check | result |
|---|---|
| C-P1a | `check_env` passes (direct, heading, obs_truth), including Gymnasium's reset-seed checks with the new seeding rule |
| C-P1b | `scarab195` -> PlaningVesselGZ / PlaningVessel (v0), `scarab195@as1` -> PlaningVesselAS / PlaningVesselASLin (as1), `probe_hullB` -> PlaningVesselGZ (v0), `probe_hullB@as1` -> PlaningVesselAS (as1); unknown key -> KeyError; `wigley10` refused; `python -m app hulls` marks which keys run |
| C-P1c | hand MPC through the app == Mission (lin) and == final_eval.make_mission (gz), seeds 0 / 3, legs 0 / 1, 40 control steps, states and commands `==` (unchanged by the fixes: the default policy settings and the env's numbers are the same) |
| C-P2a | hand and m0 through the app == `final_eval.run_mission`: all 10 metrics equal to the last bit; m0 refuses probe_hullB; `cat_p_prior` runs 2 steps; full tier: m0 30 s; `quiet_fe` puts final_eval's `log`, `out_dir`, `OPT` back |
| C-P2b | browser replay and live on both hulls after the fixes (section 4); Blender video rendered from an app record through `python -m app export ... --video ... --render` |
| C-P3 | load -> save -> load identity on all scenario files; LOTUSim waypoint syntax (robust group) |
| C-P4a | record: trajectory, sea, commands, six sensors, stamp, git commit, scenario warnings, error |
| C-P4b | fingerprint `dc3c053111da03d1` identical in three processes (new value: the identity is now part of it); save -> load identical; command replay identical on scarab195, probe_hullB and scarab195@as1; each e2e record's stamp and hull sha1 match the current code; a live run == the batch run (`a6c8c417bb9bab34`) |
| C-P5 | as before: noise off equals the truth, noise statistics in their chi-square and binomial bands, latencies exact |
| C-P6 | websocket, UDP and HTTP (with a session id) == direct env; `poses` is exactly `{header, vessels}` and its field lists equal LOTUSim's `.msg` files; server-sent events; guards incl. encoded backslash and dot forms |
| C-P7 | scarab195 (hand, heading frame) 91.0 s and probe_hullB 86.2 s complete the course, scarab195@as1 (scripted) 60.0 s -- the as1 time moved from 60.2 s because the physics track changed as1's code meanwhile (stamp `0bf993194a71 -> 108c00c5e52d`, preset `['dDt_full'] -> ['dDt_full', 'chine_exact_all']`), which `python -m app replay` of the old demo record now reports before saying DIFFERENT |
| C-R | robust group: 17 checks pass (section 1) |
| C-T | quick tier 97 s (parallel), full tier 233 s (one process at a time, low free RAM); limits 10 min / 1 h |

## 7. Review fixes (2026-10-06)

| review item | fix | check |
|---|---|---|
| path traversal through backslashes (major) | `server.safe_static` (unquote; refuse `\`, `:`, `%`, NUL, `.` / `..`; `realpath` + `commonpath` against the three folders), used by GET and HEAD | robust: 12 forms, 24 raw requests -> 404 |
| one HTTP session for all clients (major) | `lotus.SessionTable`: session id from reset, one lock per session, idle expiry, max 32; also used by UDP | robust: 4 clients x 5 concurrent steps |
| non-finite actions; policy exceptions lose the run (major) | `env._command` and `lotus.parse_cmd` refuse non-finite or non-numeric values (`ActionError`) before stepping; `runner.run` keeps the partial record with reason and traceback, `close()` in `finally`, `raise_on_error`; `LiveRun` saves it and pushes `end` | robust: env, cmd_string, websocket, UDP, HTTP; Boom policies; live run |
| `reset()` without a seed repeats the sea (major) | first reset: scenario seed; later plain resets draw from `np_random`; seed in info and header | robust: reset(11) + 3 plain resets, twice; check_env |
| waypoints not LOTUSim-compatible (major) | `waypoint_interface` with `mode` waypoints / line / circle, `[{lat, lng}]`, enabled by default; bare lists under `path` refused; section 2.3 corrected | robust: LOTUSim forms and refusals |
| `dumps` replaced text inside strings | `_clean` walks the object, `allow_nan=False` | robust |
| UDP: no retry safety, blocking resets, no eviction | `seq` (cached reply on repeat, gaps refused), worker pool, idle eviction | robust: resent step |
| validation gaps | `sensors.validate_config`; scenario numbers with key names; seeds; policies built at load; R1 warnings in the header | robust: 15 refusals, warning recorded |
| built-in policies never read the sensors | `state_source: sensors` (scripted, hand) with `app/estimator.py`; HUD "controller sees" | robust: noise changes the sensed loop, not the true-state loop; live demo on probe_hullB |
| Blender: no video, bad windows | `blender_command(clip=, mp4=)`, `find_blender`, `clip_window`, CLI `--video --render`, export without output is an argparse error | robust; one EEVEE clip rendered |
| websocket robustness | `ProtocolError` (1002 / 1007 / 1009), masks checked, fragments capped (4 MB), close then drain (no RST) | robust: four protocol violations |
| live runs unbounded | limits (4 light, 1 heavy, RAM reserve) -> 429; learned rows at rate 0; events pruned after the record, `replay` event for late readers; at most 20 finished runs in memory | robust: RAM gate, count limits; browser |
| `poses` carried `usv` | `poses` = `{header, vessels}`, extras in the reply's `usv`; ROS bridge reads `rep["usv"]` | interfaces: field lists `==` |
| records went stale silently | identity in the fingerprint; `code_changes` in replay, export, viewer | robust; old as1 demo record |
| ROS 2 bridge | doc matches argparse flags; `lotus.stamp`; timer stopped at the episode's end | read only (no ROS 2 here) |
| usability | hull listing `runnable`; `quiet_fe` context manager; caches keyed by `key_sha1`; policy labels | robust, torch |
| test gaps | `robust` group in the quick tier | 17 checks, under 30 s |

## 8. Follow-ups to the physics track's P1b (2026-10-06, "P1c" round, track A)

Asked by the physics track's notes (`learn/meta/ANYSPEED_PLAN_2026-10-05.md`
P0 "几何的使用者都从几何模块取" and "产品层" bullets, P1b review minor item 11
"隐藏状态"): the reset / replay contract with the plants' hidden state, the
viewers' hull from the exported stations, caches by `physics.cache_tag`, the
hull listing, the records refreshed for the current as1 preset. No physics
file was edited.

### 8.1 Checks written before implementation (ACCEPTANCE P4, P7, N1, N3, T)

| id | check | pass criterion |
|---|---|---|
| C-H1 | resume from any plant row | a record of `scarab195@as1` with `test_switches` = the as1 preset + `wagner_entry` (its hidden state non-trivial: F and X planes in > 20 rows): `record.plant_at(rec, j)` stepped with the recorded commands reproduces `s` and `a_cg` `==` for 40 steps, rows inside control steps and where F planes exist |
| C-H2 | resume from any control step | `record.replay(rec, start=k)`: physical fingerprint `==` the record's, k = 1, m/3, m - 2 (wagner record) and k = m/2 on every e2e demo (v0, as1, both hulls) |
| C-H3 | the plant's own log agrees | `trajectory()['hidden'][j + 1]` `==` the record's row j for every row (exact codec), `[0]` is None; restoring from it gives the same continuation |
| C-H4 | not vacuous | restoring without the hidden state, or with fresh delay lines, changes the continuation (max abs difference > 0) |
| C-H5 | nothing left out | `plantstate.audit`: after the same next step, every attribute path of the restored plant equals the original's (scratch values, logs, last-step outputs aside; tables built lazily in one plant only are listed, not compared) |
| C-H6 | the env's own state | v0 scarab195, heading mode (autopilot) and safe mode, scripted policy: `run(..., resume=(rec, k))` gives the original record's FULL fingerprint (every array, the policy's actions and the controller state) |
| C-H7 | refusals | other test switches, a format-1 record past row 0, a record of other physics code (`code_changes` named), rows out of range: refused before anything is stepped |
| C-V1 | viewer hull | viewer JSON `x_fine / yc_fine / zk_fine` `==` `hydro/hull_geometry` chine and keel (bitwise), no `taper / stem_rise` exported, `hull.cache_tag == physics.cache_tag`, `meta.hull_tag`; in the browser pane, for scarab195, planing_b@as1 and probe_hullB, the page's `ycOf / zkOf` at sample points equal the module's values (4 decimals) |
| C-V2 | caches by tag | the app's reduced-model and turn-table files and the server's viewer data are named by the hull's cache tag; a new code state writes a new file |
| C-L | hull listing | `planing_b` not runnable on v0 (v0's reason, "run it as planing_b@as1"), `planing_b@as1` runnable, no `@as1` key for displacement hulls; `UsvEnv` refuses `planing_b` with that message |
| C-N | v0 unchanged | the refreshed v0 demo records: every array present in the old and the new record bit-identical |
| C-T | cost | quick tier < 10 min, full tier < 1 h; recording the state costs < 1 % of a plant step |

### 8.2 The contract

* **Rows.** Plant row j = the plant right after plant step j (the record's
  `plant` arrays). Per row the record stores `plant['aux']` (v0's carried
  state as floats; columns in `header['state_cols']`: `prop.prime`,
  `_emerged`, `slam_count`, `peak_slam_force`, `peak_slam_force_full`,
  `capsized`, `capsize_t`, the jet's and the nozzle's delay lines) and, when
  the plant has one, `plant['hidden']`: `plant.hidden_state()` after the
  step, through an exact JSON codec (`app/plantstate.encode`: arrays keep
  dtype, shape and bytes, floats their bits, tuples stay tuples), so the app
  carries whatever the physics puts there without reading it -- a hidden
  state a later step adds travels with it. Per control step, `ctrl['ap']`
  (autopilot integrator and reference) and `ctrl['safe_state']`. Record
  format `usv-app-record/2`; header also `test_switches`, `cache_tag`,
  `state_format`, `hidden_state`, `resumed_from`.
* **Why not only `trajectory()['hidden']`.** That log exists only on the GZ
  plant built with `record=True` and holds the state BEFORE each step (None
  where a step started from the steady field); the env logs after every
  step on any plant kind. The two are checked equal (C-H3), and either can
  restore.
* **What v0 kept and nobody recorded.** The jet's and the nozzle's command
  delay lines (5 steps each on scarab195) and the jet's prime are real
  carried state of the frozen plant (`sim/actuators.Delay`,
  `sim/waterjet.Waterjet.update_prime`): a resume without them is not exact
  (C-H4). They are read and set as attributes; the frozen files are not
  touched.
* **API.** `UsvEnv(..., test_switches=)` (tests only, never from a scenario
  file); `env.resume(rec, k)` or `reset(options={"record": rec, "row": k})`;
  `record.replay(rec, start=k)`; `record.plant_at(rec, j)`;
  `runner.run(..., resume=(rec, k))`; `python -m app replay REC --start K`.
  A resume replays the sensors, the track and the reward on the recorded
  rows (no plant step; checked against the record) and restores the plant
  and the env's controller state. A policy's own memory (MPC warm start,
  learners) is not in a record and starts fresh. Strict identity (k > 0):
  the scenario sha256, the physics sha1 (code, hull, switches), the reset
  state and the sea must match, else `RecordMismatch` naming the difference
  (`code_changes`); a replay from row 0 on changed code stays allowed (it
  shows what changed).
* **Viewers.** `app/export.hull_dict` exports the plant's station arrays and
  the chine and keel from `hydro/hull_geometry` at the points
  `export.line_points` chooses (`x_fine, yc_fine, zk_fine`; since the
  review of this round, section 8.6: within `line_tol` = 0.25 mm of the
  module everywhere, the stem included, instead of 201 uniform points),
  the water shader's chine table (`x_tab, yc_tab`, at most 64 points),
  `cache_tag`, the geometry form; `taper` and `stem_rise` are no longer
  exported. `viewer/demo.html` and `viewer/proto_web.html` draw from them
  (linear between points; the mesh has a ring at every exported point; the
  shader searches its (x, yc) table); an older export without them falls
  back to the station arrays `x, yc, zk`, never to the formula. The
  deck edge follows the plant's GZ sections (keel + depth). The Blender
  renderer `viewer/proto_blender/render.py` (`hull_lines`) and
  `studies/visual_export_web.py` (`viewer/proto_web_data.json`, regenerated:
  frames and sea identical to the previous file) do the same.
  `window.__viewer.hull` exposes the lines for checks.
* **Caches.** `app/models.tag_of(hull)` = `physics.cache_tag` (the plain
  name for the legacy scarab195): `studies/_cache/app/lofi_<tag>.json`,
  `turn_<tag>_<plant>.json`. (A first version copied a fingerprint-valid
  file of an earlier tag of the same hull; removed in the review, section
  8.6: every as1 code state identifies its own model, about a minute per
  hull on first use.) The server keeps viewer data
  in `studies/_cache/app/viewer/<runs-folder sha>/<run>.<tag>.json`.

### 8.3 Files changed

`app/plantstate.py` (new), `app/env.py`, `app/record.py`, `app/runner.py`,
`app/hulls.py`, `app/models.py`, `app/export.py`, `app/server.py`,
`app/__init__.py`, `app/__main__.py`, `app/scenarios/demo_planing_b_as1.json`
(new), `studies/app_test.py` (group `resume`, e2e checks, listing),
`viewer/demo.html`, `viewer/proto_web.html` (anchored replacements by the
same script: hull block, shader chine table, console hook, note text; the
prototype's title now names the record's hull), `viewer/proto_blender/render.py`
(`hull_lines`), `studies/visual_export_web.py` (hull from
`app/export.hull_dict`), `viewer/proto_web_data.json` (regenerated),
`app/runs/demo_*` (refreshed), `viewer/demo_shots/p1c_*`.

### 8.4 Results

`python -m studies.app_test --tier full`: **166 / 166 checks pass, 167 s
wall** (7.2 GB free, light groups in parallel 88 s, torch group 79 s; log
`studies/_cache/app/test_full_p1c.log`). Quick tier: 163 / 165 on the first
run, 208 s wall (two e2e checks failed on the TEST: it rebuilt the sample
points as `x_fine + lcg`, which is not the exported points to the last bit;
fixed to the export's own `linspace`, then 18 / 18 for those groups,
`test_e2e_p1c_fix.log`).

| check | result |
|---|---|
| C-H1 | wagner record: 600 rows, F planes in 252, X in 172; plant rows 1, 234, 235, 560 (1, 6, 7, 8 steps into their control steps): 40 steps each, `s` and `a_cg` identical |
| C-H2 | wagner record from control steps 1, 16, 48 of 50; e2e demos from the middle: scarab195 (k 189 / 379), probe_hullB (179 / 359), scarab195@as1 (130 / 260), planing_b@as1 (124 / 249): all identical |
| C-H3 | 600 rows equal; row 0 None; restoring row 235 from `trajectory()['hidden'][236]` identical |
| C-H4 | without the hidden state: max abs state difference 9e-3 in 40 steps; with fresh delay lines (row 276, pending nozzle commands spread 1.4 deg): 11 |
| C-H5 | rows 235 and 200: 261 attribute paths equal after the common step; at row 200 the section table `_sec2d` and `_ycp` were built lazily in the original only (no F planes in the restored plant's step); at row 235 they exist in both and are equal |
| C-H6 | v0, 42 control steps, resumed at 17: full fingerprint identical (`ebe1f90b3b1c945a`) |
| C-H7 | all four refusals raised before stepping |
| C-V1 | tests: bitwise for all four demo hulls; the v0 formula would be off by 0.100 m (chine) and 0.069 m (keel) on planing_b@as1, 0.000 on the others (to the 0.1 mm chine floor at the stem). Browser pane (`viewer/demo_shots/p1c_*`): demo.html replays of scarab195 (v0), probe_hullB (v0), scarab195@as1, planing_b@as1, a live planing_b@as1 run (record identical to the batch run of the same code, fingerprint `c42e8eafb34142bb` both), proto_web.html on its regenerated data and on the planing_b@as1 record; the page's lines equal the module's at 8 sample points; an older export draws from its station arrays (fallback). Blender EEVEE still of planing_b@as1 from the app's scene (28 s wall): `p1c_blender_planing_b_as1_eevee.png` |
| C-V2 | files `lofi_planing_b@as1-<sha>.json`, `lofi_scarab195@as1-<sha>.json`, viewer data per tag; the old name-only files are left as they were -- **except one, rewritten by an experiment during this work: an N3 breach, section 8.5** |
| C-L | `python -m app hulls`: runnable scarab195, scarab195@as1, planing_b@as1, probe_hullB, probe_hullB@as1; planing_b: "v0 refuses it (geometry outside v0's family member, and 8 inputs v0 hard-codes): run it as planing_b@as1" |
| C-N | `demo_scarab195_hand`, `demo_hullB_hand` re-run: every array present before is bit-identical (new: `plant/aux`, `ctrl/ap`, `ctrl/safe_state`), so their fingerprints are new (`1f45dca06ca09555`, `fe506a2ab6685144`); the fingerprints quoted in sections 4 and 6 are of format-1 records |
| C-T | recording the state: 12 us (aux) + 1-6 us (hidden) per plant step against 2-5 ms for the step; the wagner record's hidden state is about 0.9 kB per row as JSON, 257 kB for the whole 12-s record compressed |

Records refreshed for the current as1 preset (`dDt_full, chine_exact_all,
kpp_strip, transom, normal_rate, verlet`): `demo_scarab195_as1_scripted`
(62.4 s to complete the course; the format-1 record of the preset
`[dDt_full, chine_exact_all]` took 60.0 s), stamp `ed7d7242efb7`;
`demo_planing_b_as1_scripted` (new; 59.8 s), stamp `f24483a0f5af`. The
other tracks edited `sim/planing_anyspeed.py` several times during this
round; between two of those code states the planing_b@as1 run was
bit-identical in every array (only the code hash moved), and each later
edit shows as `code_changes` in `python -m app replay` and the viewer.

### 8.5 Limitations and open items

* A policy's own memory is not recorded: a resumed closed-loop run equals
  the original only for policies that are functions of (t, state, track)
  -- scripted; hand MPC, m0 and learned rows continue as a branch.
* Resuming is Python / CLI only; the websocket / UDP / HTTP sessions
  (`app/lotus.py`) still reset from a scenario. A `{"op": "reset",
  "resume": {"run": ..., "row": ...}}` message would be a thin addition.
* Every as1 code edit gives as1 records a new stamp: they then replay from
  the start only (with the change reported) until re-run; refreshing them
  is a rerun of `python -m app run ...` (section 4).
* The audit compares attributes recursively to depth 6 and the first 4096
  list entries; a lazily built table present in one plant only is listed,
  not compared, at that row.
* **N3 breach (found in review).** An early experiment of this work
  rewrote the app's old name-only cache
  `studies/_cache/app/lofi_scarab195@as1.json` once (16:36): it now holds
  the 6-switch preset's model (fingerprint 35.6408); the 2-switch preset
  that wrote it originally gives 36.2912. Nothing has read the file since
  the tag-keyed names; it is renamed `lofi_scarab195@as1.stale.json` (kept,
  not deleted) so that no name-keyed reader can ever take it. The other
  old name-only file, `lofi_probe_hullB@as1.json` (02:55), was not
  touched.

### 8.6 Review fixes (P1c review of track A, 2026-10-06)

Nine review items (two major, seven minor); all fixed. The code-hash
list itself (item 2) belongs to the physics track, which added
`sim/as_kernels.py` to it at 18:13. No physics file was edited by track A.

| # | item | fix | evidence |
|---|---|---|---|
| 1 (major) | the completeness audit skipped every callable object -- the delay lines (`sim.actuators.Delay` defines `__call__`), `_geom.chine/keel/...`, `slam.wagner_slope` -- so "nothing differs" did not cover the very state the contract adds | `app/plantstate._flat`: skips only classes, modules and builtins; enters every object of sim / hydro (callable or not), a sim / hydro function's default arguments and closure cells, a bound method's instance, a partial's arguments. A reference back to either plant being compared is written as a marker: a deep copy's closures still point at the plant it was copied from (functions are not copied: `slam.wagner_slope`'s `self`), and without the marker an unaltered deep copy differed. The audit test now steps a fresh env's plant to the row instead of deep-copying it | group `resume`, new check "the audit sees callable objects": an unaltered deep copy -> `[]`; `prop.delay.buf` altered -> only `plant.prop.delay.buf[...]` named, `rudder.delay.buf` likewise; 353 paths on the wagner plant (was 261), 37 of them inside functions' defaults and closures; the audits at two rows and on the P1b plant still find nothing uncarried |
| 2 (major) | `sim/as_kernels.py` (the fk_exact kernels, now in the as1 preset) is not in `sim/physics.VERSIONS['as1']['code']`: an edit to it changes neither stamps nor cache tags | physics owner: add it to the list (not done by track A: sim/physics.py is a physics file). Track A: new test group `codehash` -- a fresh process builds the GZ plant of scarab195@as1 and the linear plant of planing_b@as1 with every implemented switch, steps each at 2 and 12 m/s, and records every repository file whose code starts (`sys.monitoring`) and every module imported; each must be in the list, or named as not numeric (only `sim/config.py`: imported, no function of it runs). A second check: `app/codeguard.load_all` imports exactly the list | fails today, as it should: `sim/as_kernels.py` runs (`FKGeo`, `FKOut`, `FKWaves`, ... under fk_exact, in the as1 preset since this round's P1c adoption) and is not in the list; every other file that runs is (`hydro/hull_geometry.py`, `hydro/planing.py`, `hydro/section2d.py`, `hydro/vessel_spec.py`, `sim/actuators.py`, `sim/physics.py`, `sim/planing_anyspeed.py`, `sim/planing_vessel.py`, `sim/planing_vessel_gz.py`, `sim/test_vessel.py`, `sim/waterjet.py`); 23 s. The hole is real, not hypothetical: `sim/as_kernels.py` changed at 17:56:47 (21 042 -> 21 432 bytes) between two e2e runs of this work (17:51-17:54 and 17:59-18:04), and scarab195@as1's cache tag was `scarab195@as1-3aa6cacfe4b8` in both. The physics track added the file to `VERSIONS['as1']['code']` at 18:13:18 (every as1 stamp and tag changed once); since then the group passes, 2 / 2 (20 s) |
| 3 (minor) | the exact codec: 0-d arrays came back with shape (1,), Fortran order was lost, object arrays were encoded as pointer bytes | `encode` keeps the shape (0-d), the memory order (C or Fortran), NaN floats by their 8 bytes; object arrays (and numpy object scalars) raise `TypeError` at encode | group `resume`, check "the exact codec": 8 cases (0-d, Fortran, bool, int8, NaN / -0.0 / inf arrays, Python floats incl. NaN, -0.0, a subnormal, a big int, numpy scalars incl. a float32 NaN, nested containers) round-trip through JSON with type, dtype, shape, order and bytes; an object array raises |
| 4 (minor) | the drawn hull near the stem: 201 uniform export points, a 141-station web mesh resampling them, a 64-point uniform shader table -- 43-46 mm (export), 52-56 mm (mesh), 77-82 mm (shader) off within a centimetre of the stem, where the chine closes like a square root | `app/export.line_points`: a uniform base of 101 points (and an offsets form's knots), then every interval whose chord leaves the chine or the keel by more than `line_tol` = 0.25 mm at any of 15 interior points is halved, until none does -- general for any form (the elliptic closure, a transom run's kink, offsets knots), deterministic. `x_tab / yc_tab`: the same from a 9-point base with the smallest tolerance (from 0.25 mm up by 1.25x) that needs at most 64 points. The web mesh has a ring at every exported point plus a uniform grid (a uniform point within 1e-9 m of an exported one is left out); the shader takes (x, yc) pairs and finds the interval by a 6-step binary search; `studies/visual_export.py` (Blender scenes) uses `line_points` too, and `render.py` puts a ring at every exported point | e2e groups, check "viewer hull": on a dense reference (20 001 uniform points plus 2 000 spaced geometrically toward the stem, down to 1e-9 L), the largest deviation from `hydro/hull_geometry` is 0.24 mm for the export and for the web mesh (272-277 stations), 0.60-0.72 mm for the shader table (61-63 points), on all four demo hulls; the previous 201 uniform points: 42.9 mm (scarab195), 45.7 mm (planing_b@as1), as the review measured. Browser pane (server restarted on this code): demo.html on planing_b@as1 -- every exported point is a mesh station, the page's `ycOf / zkOf` equal the export at all exported points, the shader table equals `x_tab / yc_tab` (padded), no console errors, renders (`viewer/demo_shots/p1c_review_planing_b_as1_side.png`, `..._chase.png`); an export without the line arrays falls back to the station arrays and a 64-point uniform table (`p1c_review_old_export_fallback.png`); proto_web.html on regenerated `viewer/proto_web_data.json` (frames, sea and meta identical to the previous file; only the hull lines changed) likewise (`p1c_review_proto_web.png`) |
| 5 (minor) | `render.py` (`hull_lines`) fell back to the v0 taper / stem_rise formula | falls back to the station arrays `x / yc / zk` closed at `x_transom / x_stem` (chine 0, keel held), as demo.html does, with a message to re-export; refuses a scene with neither; never the formula | `viewer/proto_blender/render.py:227` |
| 6 (minor) | a long-running server hashed the code at its first `import sim.physics` and compiled `sim/planing_anyspeed.py` at its first as1 plant: an edit in between made live records name code they did not run; the 8766 server served stale code silently | new `app/codeguard.py`: `load_all()` imports `sim/physics.py` and every module of every version's code list at once (the server at start-up, every `UsvEnv` -- notebooks too); `stale(ver)` says when the disk has moved on since. The server refuses new live runs (POST /api/run -> 409 StaleCode) and session resets (websocket, UDP, HTTP: an "error" reply) of a stale version, and its viewer data carry `meta.server_code_stale` (demo.html adds "server runs old physics code: restart it" to the title), set when the data are served, not when the server's cached file was written. The server's viewer cache files are now also named by `export.VIEWER_FORMAT` (2): after the hull-line change the v0 demos were still served from files of the older exporter (201 points, no shader table), cached under an unchanged legacy tag. `UsvEnv` warns once (`refuse_stale=True` raises) | group `robust`, new check: with the process's as1 hash replaced, POST /api/run of an as1 scenario -> 409, an as1 session reset -> StaleCode, a v0 session still runs; a v0 record's viewer meta says [] when fresh and names v0 while v0's hash is replaced; observed live as well: the 8766 server, started on this code before the physics track's 18:13 edit of `sim/physics.py`, answered POST /api/run of `demo_scarab195_as1.json` with 409 ("this process runs as1 physics code fe0f410063e2, the files on disk now hash to dda8bd0b6b0c: restart it ...") and its viewer data named the same; the server was restarted (18:22) (log `studies/_cache/app/server_8766.log`) and serves all four demos with `x_tab`, `code_changes` [] and `server_code_stale` [] |
| 7 (minor) | both refreshed as1 demo records were stale | re-run after the other tracks' edits: twice: at 18:05, then -- after the physics track's 18:13 edit put `sim/as_kernels.py` into the hash, which changed every as1 stamp -- at 18:22, on the code of then (the as1 preset is now `dDt_full, chine_exact_all, kpp_strip, transom, normal_rate, verlet, fk_exact, fast_path`): `demo_scarab195_as1_scripted` stamp `40ec5fac7645`, fingerprint `79519df2999f6804`, course completed in 60.96 s; `demo_planing_b_as1_scripted` stamp `955694d7fbd3`, fingerprint `8963a2fc913cdfa7`, 60.00 s; 98 and 103 s wall, of which about a minute each re-identified the reduced model under the new tags (the price of no cross-tag copy); their viewer and Blender scene files with them. The v0 demos' viewer and scene files re-exported (frames and sea identical, hull lines new); their records are unchanged, and the e2e runs of this work reproduce both bit for bit (full fingerprint). Caveat: the physics tracks were still editing; a later edit of a hashed file stales these records again (reported by `code_changes`) (`sim/as_kernels.py` included now) | `code_changes` = [] for both at 18:24 (as1 code hash on disk `085a2d43a395`); `python -m app replay ... --start 127` (scarab195@as1) and `--start 124` (planing_b@as1): IDENTICAL (21.9 s, 23.1 s); served by the restarted 8766 server with `code_changes` [] and `server_code_stale` [] |
| 8 (minor) | test gaps: safe mode never engaged in the resume test; `WAGNER_TEST` hard-coded; no resume on fk_exact / orbital / lateral_phi0 | `_switches_plus(...)` derives the switch sets from the current preset (`sim/planing_anyspeed.resolve`, its order, no duplicates): the wagner record is "the preset + wagner_entry" whatever the preset is (now `[..., verlet, fk_exact, wagner_entry, fast_path]`). New group `resume_more`: an 8-s record on the preset + all four P1b switches (hidden state at every row; plant_at at three rows inside control steps, 30 steps each; audit; replay from control steps 1, m/2, m-2), and safe mode with lowered thresholds (enter 1 deg, exit 0.8 deg, hold 1 s, the env's own SafeMode subclassed for the test) resumed where it is on with calm time 0, on with calm time > 0, and off after an entry | group `resume_more`: 408 rows, hidden state at each; plant rows 108, 205, 378 (0, 1, 6 of 12 into their control steps) identical for 30 steps; audit at row 204: 375 paths equal; replay from control steps 1, 17, 32 of 34 identical. Safe mode: on in 40 of 42 control steps, 2 entries; resumed at 2 (on, calm 0), 28 (on, calm 0.48 s), 31 (off after an entry): full fingerprint identical each time (47 s for the group) |
| 9 (minor) | listing text: planing_b's reason said twice; "8 inputs v0 hard-codes" wrong for the numerics defaults; `runnable_hull` does not see constructor refusals | REG_DOC no longer carries the clause; "and 8 other inputs differ from v0's fixed values"; `runnable_hull`'s docstring says what it does not check -- building calm plants for the listing would cost 4-7 s per key (both classes, steady attitude, righting-arm table: about 25 s for the five planing keys) and every registered and extra hull builds; an inline hull's constructor refusal is raised by `UsvEnv`, which builds its plant at once | group `robust`, "hull listing": the reason appears once, the new wording, no "v0" in the planing_b@as1 note |

Also: the reduced-model cache (`app/models.reduced_params`) no longer copies
an earlier tag's file of the same hull. The calm-water fingerprint does
separate the presets (35.6408 on the 6-switch preset, 36.2912 on the
2-switch one, 36.5027 with none, which equals v0), but a code edit that
changes only regimes the 3-second calm run never reaches (planing speed,
waves, turning) would have been copied across tags unseen. Every as1 code
state now identifies its own model, about a minute per hull on first use;
the project's name-keyed caches are read for v0 hulls only (v0's code is
frozen). The N3 breach is recorded in section 8.5.

Files changed in the review: `app/plantstate.py` (codec, audit walker),
`app/codeguard.py` (new), `app/server.py` (load at start, 409, sessions,
viewer cache name and serve-time stale flag), `app/lotus.py` (Session
`refuse_stale`), `app/env.py` (load + warn, `refuse_stale=`),
`app/export.py` (`line_points`, `line_table`, `VIEWER_FORMAT`, meta
`server_code_stale`), `app/models.py` (no cross-tag copy), `app/hulls.py`
(texts), `viewer/demo.html` and `viewer/proto_web.html` (mesh stations,
shader table, banner, `__viewer.hull.HX / XST / TAB` for checks),
`viewer/proto_blender/render.py` (`hull_lines`, `build_hull`),
`studies/visual_export.py` (`line_points`; a viewer export, not a physics
file), `viewer/proto_web_data.json` (regenerated: frames, sea and meta
identical), `studies/app_test.py` (groups `resume_more`, `codehash`, checks
in `resume`, e2e, `robust`, `interfaces`), `app/runs/demo_*` (as1 records
re-run, v0 exports re-exported), `studies/_cache/app/lofi_scarab195@as1.json`
-> `.stale.json`, `viewer/demo_shots/p1c_review_*`.

Tests (logs in `studies/_cache/app/`):
* full tier before the physics track's 18:13 edit
  (`test_full_p1c_review.log`): 175 / 176 checks, 324 s wall (4.4 GB free:
  three light groups at a time, light groups 247 s, torch 77 s); the one
  failure was `codehash` naming `sim/as_kernels.py` -- the hole of item 2,
  as intended;
* quick tier afterwards (`test_quick_p1c_review.log`): 174 / 175, 605 s
  wall (3.6 GB free: two light groups at a time; the e2e as1 groups took
  92-96 s, about a minute of it re-identifying the reduced model under the
  new tags). The one failure is a race with that edit: the test worker had
  imported `sim/physics.py` before 18:13:18 and the probe process after,
  so their lists differed by `sim/as_kernels.py`; `--groups codehash`
  re-run at 18:20: 2 / 2;
* the quick tier's 10-minute budget (T1) is met only just at 3.6 GB free;
  with the 7 GB of the previous round it took 208 s.

## 9. 冻结的仿真配置、训练时的随机化、输出规范（2026-10-09）

> 这一节说的是控制侧怎样一致地使用 2026-10-09 定下的最终仿真船（见
> `learn/meta/SIM_FINAL_2026-10-09.md`）。只改了产品层 `app/` 和它的测试，
> 没有碰 `sim/`、`hydro/` 里的任何物理文件。下文把这条船叫"最终版的船"或
> "新版船"，不用代码里的缩写来称呼它。

### 9.1 怎样选到最终版的船

* **船名。** 场景文件里 `agents.<名字>.model` 现在默认是 `scarab195@as1`。
  船名后面的 `@as1` 表示"同一条 5.4 米喷水推进滑行艇，跑在新版任意航速
  物理上"；不带后缀的 `scarab195` 仍是旧版物理，所有既往结果都在旧版上，
  旧版一个比特都没变。代码里一行就能拿到最终配置的场景：
  `app.frozen.scenario(hs=1.0, tp=5.0, duration=60, seed=0)`（迎浪、评估海况）。
* **横摇版本。** `plant` 默认仍是 `gz`，意思是"带大角度横摇、能倾覆的版本"，
  也就是评估一直在用的那个。它由 `sim.physics.gz_twin` 从船自己的物理版本
  派生，不走冻结的旧文件（旧文件不看船名，永远建旧版）。
* **步长。** 新版船默认每 0.01 秒走一步，控制器每 0.25 秒给一次指令，
  也就是一个控制步里船走 25 小步。旧版船名保持原来的 0.02 秒、12 小步，
  所以旧版的逐位复现检查都照旧通过。要报告"十分之一大值"或极值时，审查
  结论要求 0.005 秒：在场景文件 `usv.dt` 里写 0.005（或 `UsvEnv(dt=0.005)`），
  这时一个控制步是 50 小步。`usv.dt_ctrl` 可改控制步。
* **记录上的印章。** 每条记录的表头和每一步的 `info` 都带 `physics`：
  物理版本、开关组合、船名、代码哈希（`sim.physics.stamp`）。表头另有
  `physics_nominal`（名义船的印章；随机化改过船的字段时两者不同）、
  `frozen`（这回合是不是恰好冻结配置：船名、横摇版本、两个步长都对得上）、
  `randomization`（这回合抽到的随机值，没开则为空）。
* **旧记录。** 2026-10-09 之前按 0.02 秒记的新版船记录仍能回放：回放时按
  记录表头里的步长重建环境。

### 9.2 随机化怎样开、怎样关

训练时让控制器见到一族略有不同的船和海，是为了吸收最终版已知的稳态偏差
（阻力偏低一到两成、不规则波冲击加速度为实测的七成到一倍）。做法是
`app/randomize.py` 里的一个配置对象 `Randomization`：

* 它装着第 9.3 节表里每个参数的范围（上下限）和一个开关，另有一个随机
  种子 `seed` 和一个"流"的标签 `stream`（训练流或评估流，见 9.5）。
  例子：`Randomization(seed=7)` 全开；`.without("rho", "cda")` 关掉两项；
  `.only("thrust_gain")` 只留一项；也可以用字典写法放进场景文件的
  `usv.randomization`（`{"seed": 7, "mass": [1380, 1450], "rho": false}`）。
* **每回合抽一次。** `env.reset(seed=回合种子, options={"randomize": True})`
  按（配置种子、回合种子、流标签）抽一组值并应用；`options={"randomize": False}`
  或不写则不随机；`options={"randomize": 抽到的字典}` 按给定的值应用（回放
  用）。关掉某个参数不会改变其他参数抽到的值（关掉的也抽，只是丢掉）。
* **三种应用途径。** 船的字段：复制一条船改字段、给它一个新名字
  （`scarab195~rz@as1`），为它建一条新的仿真船——第一次见到这组字段时要
  7 到 9 秒（稳态航行姿态和复原力臂表要为新船算；2026-10-10 起同一组字段
  的船有缓存，再遇到只要几毫秒，另有"每几回合才换船"的选项，见 9.8）；
  海况：改有义波高、谱峰周期、浪向、谱形，并把
  波长短于 1.5 倍船长的成分幅值乘一个系数；指令：推力指令在进喷泵前乘一个
  增益（喷泵自己的上限仍在）；输出：加速度和升沉纵摇的比例因子只乘在
  观测和记录的输出上，物理一点不动，奖励也不动。
* **控制侧看到的是名义船。** 简化模型、航向自动驾驭、动作的量纲换算都用
  名义船；随机化只改被控对象。基于模型的策略直接读 `info["state"]`，那是
  真状态，输出比例因子碰不到它。
* **记录。** 表头 `randomization` 存抽到的值，`randomization_applied` 存实际
  落到船上的字段值、海况参数、被缩放的短波成分个数、三个比例因子，
  `plant_build_s` 存这回合建船花的秒数。随机化的记录照样能逐位回放：回放
  时按表头把同一组值再应用一遍（已测）。

### 9.3 参数表（默认范围；全部来自 SIM_FINAL 第 5 节）

| 名称 | 含义 | 范围 | 途径 | 依据 |
|---|---|---|---|---|
| mass | 质量，公斤 | 1340 到 1490 | 船的字段，重建船 | 数据（两次实船载况） |
| lcg | 重心纵向位置，乘名义值的倍数（对这条船即总长的 0.34 到 0.42 倍） | 0.895 到 1.105 | 船的字段，重建船 | 数据 |
| vcg | 重心高度，在名义值上加减，米 | −0.1 到 +0.1 | 船的字段，重建船 | 判断 |
| cda | 空气阻力面积，平方米 | 1.2 到 2.2 | 船的字段，重建船 | 数据（极速反算） |
| rho | 水密度，公斤每立方米 | 1000 到 1025 | 船的字段，重建船 | 数据（淡水、海水） |
| c_dc | 横流阻力系数，乘名义值的倍数 | 0.7 到 1.3 | 船的字段，重建船 | 判断（登记 E3） |
| transom_ft_crit | 船尾通气阈值（船尾吃水弗劳德数） | 2.0 到 4.5 | 按船覆盖 `hydro/vessel_spec` 的默认闭合，重建船 | 数据（三个文献来源） |
| roll_zeta | 横摇阻尼比 | 0.03 到 0.20 | 船的字段，重建船 | 判断（登记 E6） |
| cd_lat | 侧向横流阻力系数 | 0.8 到 1.2 | 船的字段，重建船 | 文献（登记 E5） |
| jet_tau | 喷泵推力时间常数，秒 | 0.4 到 0.8 | 船的字段 `jet.tau`，重建船 | 判断 |
| jet_delay | 喷泵指令延迟，秒 | 0.05 到 0.15 | 船的字段 `jet.delay`，重建船 | 判断 |
| hs | 有义波高，乘的倍数 | 0.9 到 1.1 | 海况输入 | 判断 |
| tp | 谱峰周期，乘的倍数 | 0.8 到 1.25 | 海况输入 | 数据（Fridsma 1971 周期未知） |
| direction_deg | 浪向，加减的度数 | −15 到 +15 | 海况输入 | 判断 |
| gamma | JONSWAP 谱的峰度 | 1 到 3.3 | 海况输入 | 判断 |
| short_wave | 波长短于 1.5 倍船长的成分，幅值乘的倍数 | 0.6 到 1.4 | 海况输入 | 数据（短波升沉偏大） |
| thrust_gain | 推力指令到实际推力的增益 | 0.85 到 1.15 | 指令扰动 | 数据（静水阻力偏低） |
| acc_scale | 重心和船首加速度输出同乘的倍数 | 0.7 到 1.3 | 输出扰动 | 数据（不规则波） |
| motion_scale | 升沉、纵摇及其速率在观测里乘的倍数 | 0.8 到 1.2 | 输出扰动 | 数据（规则波） |

全部 19 项都有现成的旋钮，没有一项被落下。明说做不到的两件事（和
SIM_FINAL 一致）：静止时偏大的附加质量和缺失的辐射阻尼没有开关；折角线
浸湿后的截面力没有开关，由 `acc_scale` 代替。另外两个字段按 SIM_FINAL 的
建议不随机：附加质量系数、浮力折减。旧版船上船尾通气阈值没有对应物
（旧版没有船尾状态），若在旧版船上开它，记录表头的 `unapplied` 会写明。

### 9.4 输出规范

* **观测里新增 `accel`。** 两个数：重心和船首测点的垂向加速度，单位 g，
  先经过"二阶巴特沃斯低通、截止 20 赫兹（全尺度）"的因果形式（在线逐步
  计算，起始状态取第一个样本的稳态，没有起步瞬态），再乘上 `acc_scale`。
  滤波器系数用闭式公式算出，和 scipy 的结果相差在小数点后十四位（已测）。
  场景 `usv.filter_hz` 可改截止频率；缩到船模尺度时乘根号下（全尺度船宽
  除以模型船宽）。
* **船首测点。** 船首柱后十分之一船长处，由重心值和船首站值线性内插
  （`app.accel.bow_gauge`），用竖直分量。记录表头存了测点和船首站的体坐标
  （`x_bow_gauge`、`x_bow_station`）。
* **记录里。** 原始值照旧保留：`a_cg`、`a_bow`（米每二次方秒，船首站值），
  新增 `acc_obs`（观测看到的两个滤波后的数）。离线统计一律用零相位滤波
  （`app.accel.record_stats(记录)`，内部是 scipy 的 filtfilt），去掉运行的
  前十分之一。
* **统计量的定义**（`app/accel.py`，和审查脚本 `studies/derive_v6_substep.py`
  一致，脚本只读不改）：规则波取每个遭遇周期的最大值再平均；不规则波取
  "正峰均值"（每次正向越零到回零之间取一个最大值再平均，Fridsma 1971 的
  读法；取峰函数与 `studies/exp_planing.excursion_peaks` 逐个相等，已测）、
  "十分之一大值"（最高十分之一的峰的平均）、最大值、峰的个数；事件计数
  （例如每小时超过某个 g 的峰数）也在这些峰上数。
* **奖励不变。** 奖励里用的仍是未滤波的重心加速度，和既往任务定义逐位相同。
* **其他观测。** `motion_scale` 乘在每个传感器看到的状态的升沉、纵摇及
  其速率上，以及 `obs_truth` 的真值观测上；惯性测量单元的比力因此跟随
  升沉的比例因子（不是加速度因子）。

### 9.5 种子

* 评估用固定的一组回合种子（`learn/repro/task.EVAL_SEEDS`，0 到 7），
  海况种子等于回合种子，这与评估脚本一致。
* 训练流里，`reset()` 不给种子时自动抽的回合种子永远不会落在评估种子上
  （抽到就重抽；三千次抽样无一命中，已测）。显式传入评估种子并要求随机化
  时，环境给出警告。
* 随机化本身的随机数由（配置种子、回合种子、流标签）决定：同一个回合种子
  在训练流和评估流抽到的值不同，同一流、同一种子两次抽到的值和整段运行
  逐位相同（已测）。

### 9.6 测试与结果（`python -m studies.app_randomization_test`，30 项全部通过，170 秒）

* 三个检查组：`frozen`（默认配置、步长、Gym 的 `check_env`、印章、滤波器
  与统计函数）、`randomization`（三回合开随机化加一回合关：输出有限、抽样
  进 info 和表头、船的副本和它自己的印章；关随机化、固定种子时两次新建
  环境的前 50 个控制步逐位相同；开随机化、同种子两次相同；随机化记录逐位
  回放；输出比例因子不碰物理；短波成分缩放准确；配置写法与拒绝；训练流
  避开评估种子）、`reference`（下面的参考数）。
* **单步耗时**（本机单核，最终版的船、可倾覆版本、步长 0.01 秒，20 节
  附近）：仿真船自身 5.4 到 6.9 毫秒一步（随航速和机器当时的负载变）；
  环境本身在此之上加约 0.3 毫秒（不带传感器）到 0.5 毫秒（带默认的六个
  传感器），各次实测的整体数在 7.2 到 7.9 毫秒之间，即一个 0.25 秒的控制步
  约 0.19 秒，比实时快约三成。随机化改了船的字段时，每回合建船另加 4 到
  5 秒（三回合实测 5.4、4.8、4.5 秒）；只随机海况、指令增益、输出因子时
  不重建，几毫秒。
* **20 节迎浪三级海况的参考数**（有义波高 1.0 米、谱峰周期 5 秒、60 秒、
  步长 0.01 秒、测试里用一个简单的比例加积分油门把航速稳在 20 节附近，
  实测均值 19.7 节，范围 16.7 到 21.4 节；统计去掉前 6 秒；结果存在
  `studies/_cache/app/reference_20kn_head_ss3.json`）：滤波后重心正峰均值
  0.95 g，十分之一大值 3.2 g，最大 4.9 g，59 个峰；船首测点正峰均值 2.7 g，
  十分之一大值 8.0 g，最大 9.7 g；不滤波的正峰均值重心 1.0 g、船首 3.1 g；
  重心超过 1 g 的峰约每小时 1270 个。十分之一大值和极值按规范应以 0.005 秒
  步长重跑（`--full` 选项会加跑）。
* 既往的产品层测试（`python -m studies.app_test --tier quick`，日志
  `studies/_cache/app/test_quick_randomization_round.log`）：175 项中 173 项
  通过，281 秒（空闲内存 7.3 GB，并行）。两项未通过：一项是本轮引起的
  （"代码是否改过"的判断改用了名义船的印章，于是测试里故意改坏记录印章
  的检查没被点名），已改回"名义船上的回合比自己的印章，随机化改过船的
  回合才比名义印章"，`robust` 组重跑 18 项全部通过；另一项与本轮无关、在
  本轮之前就已存在：仿真船在 2026-10-07 新加了一个闭包函数 `_esm`（记住
  船宽随纵向位置的函数，闭包里引用了船的几何对象），"清点"检查在走深拷贝
  时顺着这个函数走到了原船的几何对象，于是把一份未改动的深拷贝报成有差别
  （在旧的 0.02 秒步长上同样复现，第一个控制步后就出现）。修法应在
  `app/plantstate.py` 的遍历器里：函数闭包和默认参数里若引用的是仓库自己的
  对象（有属性字典的对象），只写一个类型标记而不再深入——这些对象都会从
  船的属性走到。本轮按范围限制没有改它，留给负责恢复与清点机制的一轮
  （已于 2026-10-10 修好，见 9.8）。
  四个端到端演示（含两条新版船的演示，现在按 0.01 秒步长跑，各约 195 秒）、
  等价性、回放、传感器、接口、恢复的其余检查、代码哈希、torch 组全部通过。

### 9.7 改动的文件

新增 `app/frozen.py`（冻结配置的常量和场景构造器）、`app/randomize.py`
（配置、抽样、应用）、`app/accel.py`（滤波、测点、统计）、
`studies/app_randomization_test.py`；修改 `app/env.py`（默认步长按版本、
随机化作为 reset 选项、输出滤波通道、记录和表头的新字段、回放时重放抽样）、
`app/scenario.py`（默认船名、`usv.dt`、`usv.dt_ctrl`、`usv.filter_hz`、
`usv.randomization`）、`app/record.py`（按记录的步长回放；代码变动的判断用
名义船的印章）、`app/runner.py`（`run(..., randomize=)` 透传）、
`app/__init__.py`（模块清单）。`sim/`、`hydro/` 未改。

### 9.8 两处修补：清点检查遇到函数、随机化回合的建船开销（2026-10-10）

> 只改了 `app/plantstate.py`、`app/randomize.py`、`app/env.py` 和两份测试，
> `sim/`、`hydro/` 一个文件没碰。记录里既有的数组和表头字段一个比特没变，
> 只在表头新增了下面说的几个字段。

**修补一：清点检查遇到函数时只写标记。** 9.6 节末尾那项早就存在的失败
（"恢复"组里"清点检查能看见可调用对象"一项）修好了。起因复述一遍：
仿真船从 2026-10-07 起保存一个闭包函数（记住船宽随纵向位置的函数），
闭包里引用船自己的几何对象；Python 的深拷贝不复制函数，所以拷贝出来的
船里这个函数仍指向原船的几何对象。清点遍历器顺着拷贝的函数走进原船的
几何对象，走出了一条原船自己没有的路径（原船的几何对象早已在 `_geom`
下面记过，不再重复记），于是一份未改动的深拷贝被报成 29 处差别。做法：
遍历器遇到函数、绑定方法或偏函数时，不再进入默认参数、闭包、所绑定的
对象，只写一个稳定的标记（函数所在模块、限定名、代码的哈希，嵌套的代码
也算进去）。后果写在模块说明里：只能经过函数的闭包才摸得到的对象不再
被清点；今天船里这类对象都另有属性路径（几何对象就是 `_geom`，龙骨静止
线在船尾的命名空间里，闭包里的 self 就是船本身），将来若有闭包抓住一个
别处摸不到的可变对象，要把它暴露成属性才会被清点。把一个函数换成另一个
仍会被点名（标记不同）；定义了 `__call__` 的对象（延迟线、滤波器）不是
函数，照旧逐项清点。测试：未改动的深拷贝报零差别；改延迟线被点名；换掉
一个函数被点名；另建一条船，它自己的闭包的标记与原船的相同。

**修补二：随机化回合的建船开销。**

* **剖析（a）。** 一次抽到船的字段的重置在本机空闲时约 9 秒，拆开是：
  设计航速下的稳态航行姿态约 3.9 秒（建基础船时算）、复原力臂表约 2.0 秒
  （建大角度横摇版本时算）、起步航速（设计航速的八成）下的稳态姿态约
  3.2 秒（取初始状态时算；这一项不在表头的 `plant_build_s` 里，所以 9.6 节
  只报了 4 到 5 秒）。三项都在仿真船内部，前两项本来就有缓存：稳态姿态按
  船的清单键控，同一组字段再抽到就命中；复原力臂表按船对象的内存地址键控。
  原先每回合都深拷贝出一个新的船对象，复原力臂表因此每回合必算，同一组
  字段再抽到也要 2 到 3 秒。同一个船对象再建一次只要几毫秒（量过：重建
  6 毫秒，从样板船换海况 3 毫秒，300 步与冷建的船逐位相同）。
* **每几回合才换船（b）。** `Randomization` 新增 `hull_every`（默认 1）。
  设为 N 时，船的字段只在这个环境的第 0、N、2N……个随机化回合重新抽并
  重建船，中间的回合沿用当前这条船，海况、推力增益、输出比例因子仍每回合
  自己抽。随机数流不动：每回合照旧按固定顺序抽全部参数，被跳过的回合抽到
  的船字段丢掉，和关掉某个参数时一样，所以开不开这个选项，其他参数抽到的
  数一个不变（已测）。回合实际跑的那组值（`info` 和表头的 `randomization`）
  里写的是当前生效的船字段，所以记录仅凭表头就能回放（已测，逐位相同）；
  表头和 `info` 新增 `randomization_hull_from`，说明生效的船是哪个回合抽的
  （抽它的回合种子、它在本环境随机化回合里的序号、本回合的序号、N）。
  关随机化的回合和给定抽样字典的回合既不计数也不换船；换配置
  （`reset` 的 `randomization` 选项）从头计数。
* **船的缓存（c）。** `Randomization` 新增 `plant_cache`（默认 8，0 为关）。
  环境把为随机化的船建好的仿真船留在一个按"船字段及其值"键控的最近最少
  使用表里，每项存船对象和一条从未走过一步的样板船。每个回合——无论命中
  还是未命中——都从样板船按本回合的海况另造一条新船（仿真船自己的
  `with_sea`，同一个船对象，所以仿真船内部的两张表都命中），绝不把样板船
  本身交给回合，回合之间因此不可能互相渗漏状态。验证：同一条缓存的船跑两个
  不同海况种子的回合，与新环境单独跑出的结果逐位相同；再跑第一个回合也与
  第一次逐位相同；关掉缓存逐回合冷建，结果也逐位相同（缓存不改任何比特）。
  `hull_every` 生效的那条船在缓存关掉时也留着。表头新增 `plant_cache`：
  本回合的船来自名义船、新建、还是缓存，以及表的当前项数和上限。
* **可跳过的表（d）：没有。** 三项全被最终配置用到：复原力臂表是横摇恢复
  力矩的来源，设计航速姿态用于侧向附加质量与阻尼，起步航速姿态就是初始
  状态。`sim/` 未改。
* **开销（本机实测，随机器当时负载变；测试组 `hull_every` 最后一项每次
  都量）。** 抽到船字段、缓存未命中：一次重置 6 到 9 秒（两次测试 5.9、
  6.5 秒，空闲时的剖析 9.0 秒）；命中：平均 6 到 7 毫秒，最大 11 毫秒；
  `hull_every` 取 10 时摊到每回合 0.6 到 0.65 秒；只随机海况、增益、输出：
  8 到 9 毫秒，与关随机化一样（一个进程里名义船的第一次重置另有一次约
  3 秒的起步航速稳态计算，之后不再有）。改之前每个抽到船字段的回合都是
  6 到 9 秒。
* **内存。** 仿真船为每条建过的船永久保留船对象和复原力臂表（一条约
  17 KB 加船对象本身），产品层不能释放；长训练里每建一万条不同的船约多占
  两三百兆。`hull_every` 和缓存减少建船次数，是控制这项增长的办法。
* **记录格式。** 表头新增 `randomization_hull_from`、`plant_cache`，
  `randomization_config` 多了 `hull_every`、`plant_cache` 两项，`info` 新增
  `randomization_hull_from`；指纹不含这些字段，既有记录照旧回放。
* **测试。** `python -m studies.app_randomization_test`（新增 `hull_every`
  组：缓存表本身、六回合的 hull_every=3、跳过换船的回合回放、缓存隔离、
  关缓存同比特、重置开销）：全套 38 项，211 秒；首轮 37 项通过，唯一未过的是新写的缓存表单元测试里数错了命中次数的一句断言（缓存本身无误），改正后 `hull_every` 组单独重跑 8 项全部通过，57 秒。
  `python -m studies.app_test --groups resume,resume_more`（修补一直接
  相关的两组）：17 项全部通过，110 秒（原先失败的那项现在通过）。按指示未重跑完整的 quick 层和四个
  端到端演示。
