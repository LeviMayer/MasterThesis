#!/usr/bin/env python3
# Console waypoint navigation + LiDAR wall/corner intelligence + recording (go_stanford)
#
# Ground plane: (x, y) because translation is (x,y,z) and z is height.
#
# Improvements over plain FTG:
# - WALL-FOLLOW mode when "hugging" a wall (keeps you moving, helps around corners)
# - CORNER ESCAPE when front+one side is tight (pulls you around the corner)
# - Better DOOR attractor: prefers far+wide openings in FRONT sector
# - STILL: Turn-lock, stuck recovery, door-commit, always-blend goal+avoid
#
# Webots: robot controller must be <extern>
# Keys (Webots 3D view focus):
#   R toggle recording
#   P pause
#   Q quit

import os, sys, time, math, csv, random, threading
from datetime import datetime
from queue import Queue, Full, Empty
from threading import Thread

import numpy as np
from PIL import Image as PILImage

WEBOTS_HOME = os.getenv("WEBOTS_HOME", "/usr/local/webots")
api_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from controller import Supervisor, Keyboard

# -------- Device names ----------
CAMERA_NAME = "camera"
LIDAR_NAME  = "lidar"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_MOTOR_NAME  = "left wheel motor"

USE_SELF_NODE = True
ROBOT_DEF_NAME = "TurtleBot3Burger"

# -------- Output ----------
GO_ROOT = "train/vint_train/data/data_splits/go_stanford"
SAVE_EVERY_N_STEPS = 2
JPEG_QUALITY = 85
QUEUE_MAX = 256

# -------- Diff-drive tuning ----------
MAX_WHEEL_SPEED = 6.28
V_GAIN = 0.90 * MAX_WHEEL_SPEED
W_GAIN = 0.60 * MAX_WHEEL_SPEED
SMOOTH = 0.20

# -------- Waypoint navigation ----------
GOAL_TOL = 0.18
GOAL_SLOW_DIST = 0.60
GOAL_W_GAIN = 1.05
GOAL_MIN_V = 0.10
GOAL_MAX_V = 0.85

STEER_SIGN = +1  # flip if mirrored

# -------- LiDAR preprocessing ----------
MAX_RANGE_CAP = 8.0
MIN_VALID = 0.05
SMOOTH_WINDOW = 7

# Safety (earlier reaction)
BUBBLE_RADIUS_M = 0.22
SAFE_FRONT = 0.40
SLOW_FRONT = 0.95

# -------- Door seeking ----------
DOOR_MIN_DIST = 0.70
DOOR_MIN_WIDTH_DEG = 6.0
DOOR_BIAS_AFTER_SEC = 6.0
DOOR_SCORE_BOOST = 3.0
DOOR_FRONT_SPAN_DEG = 120.0

# Door commit
DOOR_COMMIT_SEC = 1.2
DOOR_COMMIT_MAXW = 0.60
DOOR_COMMIT_V = 0.45

# Narrow corridor mode
NARROW_LR_THRESH = 0.40
NARROW_SPEED = 0.35

# Stuck recovery
STUCK_WINDOW_SEC = 2.5
STUCK_DIST_M = 0.08
ESCAPE_TURN_SEC = 1.2

# Turn-lock hysteresis
TURN_LOCK_SEC = 0.8
TURN_LOCK_FRONT_CLEAR = 0.75
TURN_LOCK_EPS_SIDE = 0.07

# Deadband
DEADBAND_DEG = 4.0

# Always-blend
BLEND_LAMBDA_MIN = 0.25
BLEND_LAMBDA_MAX = 0.95

# -------- NEW: Wall-follow + corner escape ----------
WALL_NEAR = 0.45          # if one side closer than this => wall-follow
WALL_TARGET = 0.55        # try to keep this distance from wall
WALL_FOLLOW_W = 0.35      # steering strength in wall-follow
WALL_FOLLOW_V = 0.55      # speed in wall-follow

CORNER_FRONT = 0.45       # if front closer than this => corner logic
CORNER_SIDE = 0.35        # side threshold for corner
CORNER_TURN = 0.85        # strong turning to pull around
CORNER_V = 0.10

random.seed(0)

def clamp(x, lo, hi): return max(lo, min(hi, x))

def capture_image(camera) -> PILImage.Image:
    data = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (w, h), data).convert("RGB")

def yaw_from_node_orientation_xy(node) -> float:
    m = node.getOrientation()
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    fx = m00
    fy = m10
    return float(math.atan2(fy, fx))

def make_run_folder(go_root: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(go_root, "manual_recordings")
    traj_dir = os.path.join(base, f"console_wp_{run_id}")
    img_dir = os.path.join(traj_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    traj_csv = os.path.join(traj_dir, "traj.csv")
    return traj_dir, img_dir, traj_csv

def writer_worker(q: Queue, stop_flag: dict):
    while not stop_flag["stop"]:
        try:
            item = q.get(timeout=0.1)
        except Empty:
            continue
        if item is None:
            break
        img_path, pil_img = item
        try:
            pil_img.save(img_path, format="JPEG", quality=JPEG_QUALITY, optimize=False)
        except Exception:
            pass
        finally:
            q.task_done()

def drive_diff(v_cmd, w_cmd, motors):
    v_cmd = clamp(v_cmd, -1.0, 1.0)
    w_cmd = clamp(w_cmd, -1.0, 1.0)

    v = V_GAIN * v_cmd
    w = W_GAIN * w_cmd

    if abs(v_cmd) > 0.1:
        w = clamp(w, -0.9*abs(v), 0.9*abs(v))

    v_l = clamp(v - w, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    v_r = clamp(v + w, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    motors[0].setVelocity(v_r)  # right
    motors[1].setVelocity(v_l)  # left

def moving_average(x: np.ndarray, win: int):
    if win <= 1: return x
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, k, mode="valid").astype(np.float32)

def preprocess_ranges(ranges: np.ndarray):
    r = ranges.astype(np.float32, copy=True)
    r[~np.isfinite(r)] = MAX_RANGE_CAP
    r[(r < MIN_VALID) | (r > MAX_RANGE_CAP)] = MAX_RANGE_CAP
    r = np.clip(r, MIN_VALID, MAX_RANGE_CAP)
    r = moving_average(r, SMOOTH_WINDOW)
    return r

def sectors_min(r: np.ndarray, mid: int):
    n = r.size
    fw = max(10, n // 6)  # wide front band
    lo = max(0, mid - fw)
    hi = min(n, mid + fw + 1)
    front = r[lo:hi]

    left_half = r[mid+fw: mid + n//2] if (mid + n//2) < n else r[mid+fw:]
    right_half = r[mid - n//2: mid-fw] if (mid - n//2) >= 0 else r[:mid-fw]

    L = float(left_half.min()) if left_half.size else float(r.min())
    R = float(right_half.min()) if right_half.size else float(r.min())
    F = float(front.min()) if front.size else float(r.min())
    return L, F, R

def apply_bubble(r: np.ndarray, angle_per_index: float):
    i_min = int(np.argmin(r))
    d_min = float(r[i_min])
    if d_min <= 0.001:
        return r
    theta = math.asin(clamp(BUBBLE_RADIUS_M / d_min, 0.0, 1.0))
    bubble_idx = int(theta / angle_per_index) if angle_per_index > 1e-6 else 6
    bubble_idx = max(4, bubble_idx)
    rr = r.copy()
    lo = max(0, i_min - bubble_idx)
    hi = min(r.size, i_min + bubble_idx + 1)
    rr[lo:hi] = MIN_VALID
    return rr

def find_gaps(mask: np.ndarray):
    gaps = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        gaps.append((i, j-1))
        i = j
    return gaps

def choose_ftg(rr: np.ndarray, mid: int, angle_per_index: float, span_deg: float = 100.0):
    n = rr.size
    span = int((math.radians(span_deg) / max(angle_per_index, 1e-6)))
    lo = max(0, mid - span)
    hi = min(n, mid + span + 1)
    idx = np.arange(lo, hi, dtype=np.int32)
    ang_pen = np.abs(idx - mid) / float(span + 1)
    score = rr[lo:hi] - 0.6 * ang_pen
    return int(idx[int(np.argmax(score))])

def choose_door(rr: np.ndarray, mid: int, angle_per_index: float, span_deg: float):
    """
    Improved door choice:
    - consider only front sector
    - candidates = contiguous indices where rr > DOOR_MIN_DIST
    - score = width_deg * (mean_dist + 0.5*max_dist)  - turn_pen
    """
    n = rr.size
    span = int((math.radians(span_deg) / max(angle_per_index, 1e-6)))
    lo = max(0, mid - span)
    hi = min(n, mid + span + 1)
    rr_front = rr[lo:hi]

    mask = rr_front > DOOR_MIN_DIST
    gaps = find_gaps(mask)
    if not gaps:
        return None

    best_i, best_score = None, -1e9
    for a, b in gaps:
        width_idx = b - a + 1
        width_deg = width_idx * (angle_per_index * 180.0 / math.pi)
        if width_deg < DOOR_MIN_WIDTH_DEG:
            continue
        seg = rr_front[a:b+1]
        mean_d = float(np.mean(seg))
        max_d  = float(np.max(seg))
        cand_local = (a + b) // 2
        cand = lo + cand_local
        turn_pen = 2.0 * (abs(cand - mid) / float(n))
        score = width_deg * (mean_d + 0.5 * max_d) - 3.0 * turn_pen
        if score > best_score:
            best_score = score
            best_i = cand

    return best_i

def controller_ftg_door(ranges: np.ndarray, mid: int, angle_per_index: float, door_bias: float):
    r = preprocess_ranges(ranges)
    L, F, R = sectors_min(r, mid)

    # emergency
    if F < SAFE_FRONT:
        v = 0.0
        w = +1.0 if L > R else -1.0
        return v, w, L, F, R, False

    rr = apply_bubble(r, angle_per_index)

    ftg_i = choose_ftg(rr, mid, angle_per_index, span_deg=110.0)
    door_i = choose_door(rr, mid, angle_per_index, span_deg=DOOR_FRONT_SPAN_DEG)

    used_door = False
    target_i = ftg_i
    if door_i is not None and door_bias > 0.0:
        if rr[door_i] * DOOR_SCORE_BOOST * door_bias >= rr[ftg_i]:
            target_i = door_i
            used_door = True

    target_angle = (target_i - mid) * angle_per_index
    if abs(target_angle) < math.radians(DEADBAND_DEG):
        w_cmd = 0.0
    else:
        w_cmd = clamp((target_angle / (math.pi/2)), -1.0, 1.0) * STEER_SIGN

    v_cmd = 0.85
    if F < SLOW_FRONT:
        v_cmd = 0.45
    v_cmd *= clamp(1.0 - 0.55*abs(w_cmd), 0.30, 1.0)

    # narrow corridor
    if (L < NARROW_LR_THRESH) and (R < NARROW_LR_THRESH) and (F > SAFE_FRONT):
        v_cmd = min(v_cmd, NARROW_SPEED)
        w_cmd = clamp(w_cmd, -0.55, 0.55)

    return v_cmd, w_cmd, L, F, R, used_door

def go_to_goal_controller_xy(x, y, yaw, gx, gy):
    dx = gx - x
    dy = gy - y
    dist = math.hypot(dx, dy)

    desired = math.atan2(dy, dx)
    err = desired - yaw
    while err > math.pi: err -= 2*math.pi
    while err < -math.pi: err += 2*math.pi

    if abs(err) < math.radians(3.0):
        w = 0.0
    else:
        w = clamp((GOAL_W_GAIN * err) / (math.pi/2), -1.0, 1.0)
    w *= STEER_SIGN

    if dist < GOAL_SLOW_DIST:
        v = clamp(dist / GOAL_SLOW_DIST, 0.0, 1.0) * 0.45 + GOAL_MIN_V
    else:
        v = GOAL_MAX_V

    v *= clamp(1.0 - 0.55*abs(w), 0.35, 1.0)
    v = clamp(v, 0.0, 1.0)
    return v, clamp(w, -1.0, 1.0), dist

# -------- Console goal input thread --------
goal_lock = threading.Lock()
goal_queue = []

def console_reader():
    print("Console goals: enter `x y` (e.g. 1.2 -0.6).")
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            xs, ys = line.split()
            gx, gy = float(xs), float(ys)
            with goal_lock:
                goal_queue.append((gx, gy))
        except Exception:
            print("Invalid input. Format: x y")

def pop_next_goal():
    with goal_lock:
        if goal_queue:
            return goal_queue.pop(0)
    return None

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    if USE_SELF_NODE:
        node = robot.getSelf()
    else:
        node = robot.getFromDef(ROBOT_DEF_NAME)
        if node is None:
            raise RuntimeError(f"Robot DEF '{ROBOT_DEF_NAME}' not found.")
    translation_field = node.getField("translation")

    cam = robot.getDevice(CAMERA_NAME)
    cam.enable(timestep)

    lidar = robot.getDevice(LIDAR_NAME)
    lidar.enable(timestep)
    try:
        lidar.enablePointCloud(False)
    except Exception:
        pass

    motors = [robot.getDevice(RIGHT_MOTOR_NAME), robot.getDevice(LEFT_MOTOR_NAME)]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    kb = Keyboard()
    kb.enable(timestep)

    n = int(lidar.getHorizontalResolution()) if hasattr(lidar, "getHorizontalResolution") else 400
    fov = float(lidar.getFov()) if hasattr(lidar, "getFov") else (2*math.pi)
    mid = n // 2
    angle_per_index = fov / max(1, (n - 1))

    traj_dir, img_dir, traj_csv = make_run_folder(GO_ROOT)
    f = open(traj_csv, "w", newline="")
    wcsv = csv.writer(f)
    wcsv.writerow(["t","img","x","y","z_h","yaw","mode","gx","gy","v_cmd","w_cmd","L","F","R","lam","note"])

    q = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}
    Thread(target=writer_worker, args=(q, stop_flag), daemon=True).start()
    Thread(target=console_reader, daemon=True).start()

    print("===================================")
    print("Console Waypoint (x,y) + LiDAR Wall/Corners + Recording")
    print(f"Save dir: {traj_dir}")
    print("Keys: R rec | P pause | Q quit")
    print("Console: type `x y`")
    print("===================================")

    recording = False
    paused = False

    last_progress_check = time.time()
    last_pos = translation_field.getSFVec3f()
    door_bias = 0.0
    escape_until = 0.0

    turn_lock_dir = 0
    turn_lock_until = 0.0

    door_commit_until = 0.0
    door_commit_w = 0.0

    mode = "WAIT_GOAL"
    gx = gy = None

    v_s, w_s = 0.0, 0.0
    step = 0
    t0 = time.time()

    try:
        while robot.step(timestep) != -1:
            step += 1
            now = time.time()

            # keys
            key = kb.getKey()
            while key != -1:
                if key in (ord("R"), ord("r")):
                    recording = not recording
                    print(f"[INFO] recording={recording}")
                elif key in (ord("P"), ord("p")):
                    paused = not paused
                    print(f"[INFO] paused={paused}")
                elif key in (ord("Q"), ord("q")):
                    print("[INFO] quit")
                    return
                key = kb.getKey()

            if paused:
                drive_diff(0.0, 0.0, motors)
                continue

            # pick goal if waiting
            if mode == "WAIT_GOAL":
                g = pop_next_goal()
                if g is not None:
                    gx, gy = g
                    mode = "GOAL"
                    print(f"[GOAL] x={gx:.3f}, y={gy:.3f}")
                else:
                    drive_diff(0.0, 0.0, motors)
                    # record while waiting
                    if recording and (step % SAVE_EVERY_N_STEPS == 0):
                        t = time.time() - t0
                        x, y, z_h = translation_field.getSFVec3f()
                        yaw = yaw_from_node_orientation_xy(node)
                        img_name = f"{step:08d}.jpg"
                        img_path = os.path.join(img_dir, img_name)
                        try:
                            q.put_nowait((img_path, capture_image(cam)))
                        except Full:
                            pass
                        wcsv.writerow([f"{t:.3f}", img_name, f"{x:.6f}", f"{y:.6f}", f"{z_h:.6f}",
                                       f"{yaw:.6f}", "WAIT", "", "", "0.000","0.000","","","","","WAIT"])
                    continue

            # sensors
            ranges = np.array(lidar.getRangeImage(), dtype=np.float32)
            if ranges.size != n:
                n = ranges.size
                mid = n // 2
                angle_per_index = fov / max(1, (n - 1))

            # pose
            x, y, z_h = translation_field.getSFVec3f()
            yaw = yaw_from_node_orientation_xy(node)

            # progress / stuck
            if mode == "GOAL" and (now - last_progress_check) >= STUCK_WINDOW_SEC:
                x0, y0, z0 = last_pos
                dist_prog = math.hypot(x - x0, y - y0)
                if dist_prog < STUCK_DIST_M:
                    escape_until = now + ESCAPE_TURN_SEC
                    door_bias = 1.0
                else:
                    door_bias = max(0.0, door_bias - 0.20)
                last_pos = (x, y, z_h)
                last_progress_check = now

            r_dbg = preprocess_ranges(ranges)
            Lm, Fm, Rm = sectors_min(r_dbg, mid)

            v_goal, w_goal, dist_to_goal = go_to_goal_controller_xy(x, y, yaw, gx, gy)

            if dist_to_goal < GOAL_TOL:
                print(f"[ARRIVED] x={gx:.3f}, y={gy:.3f}")
                mode = "WAIT_GOAL"
                v_t, w_t = 0.0, 0.0
                lam = 0.0
                note = "ARRIVED"
            else:
                note = ""

                # escape
                if now < escape_until:
                    escape_dir = +1.0 if Lm > Rm else -1.0
                    v_t, w_t = 0.10, escape_dir
                    lam = 1.0
                    note = "ESCAPE"
                else:
                    # corner escape (NEW): front tight + one side tight => pull around corner
                    if (Fm < CORNER_FRONT) and (min(Lm, Rm) < CORNER_SIDE):
                        # if right is tighter, turn left; if left is tighter, turn right
                        w_t = +CORNER_TURN if (Rm < Lm) else -CORNER_TURN
                        v_t = CORNER_V
                        lam = 1.0
                        note = "CORNER"
                    else:
                        # wall-follow (NEW): keep moving when hugging a wall
                        if (Lm < WALL_NEAR) ^ (Rm < WALL_NEAR):  # exactly one side near
                            # follow the nearer wall
                            if Lm < WALL_NEAR:
                                # left wall: steer slightly right if too close, else slightly left
                                err = clamp((WALL_TARGET - Lm), -0.30, 0.30)
                                w_wall = clamp(-WALL_FOLLOW_W * err / 0.30, -0.6, 0.6)
                            else:
                                # right wall
                                err = clamp((WALL_TARGET - Rm), -0.30, 0.30)
                                w_wall = clamp(+WALL_FOLLOW_W * err / 0.30, -0.6, 0.6)

                            # still avoid if front is getting closer
                            v_avoid, w_avoid, L, F, R, used_door = controller_ftg_door(
                                ranges=ranges, mid=mid, angle_per_index=angle_per_index,
                                door_bias=1.0 if (now - t0) > DOOR_BIAS_AFTER_SEC else 0.0
                            )

                            # blend wall-follow with avoidance
                            alpha = clamp((SLOW_FRONT - Fm) / max(SLOW_FRONT - SAFE_FRONT, 1e-6), 0.0, 1.0)
                            lam = clamp(0.35 + 0.60 * alpha, 0.35, 0.95)

                            v_t = (1.0 - lam) * WALL_FOLLOW_V + lam * v_avoid
                            w_t = (1.0 - lam) * w_wall + lam * w_avoid
                            note = "WALL"
                        else:
                            # turn-lock (kept)
                            if (turn_lock_dir == 0) and (Fm < SAFE_FRONT):
                                if abs(Lm - Rm) < TURN_LOCK_EPS_SIDE:
                                    turn_lock_dir = random.choice([+1, -1])
                                else:
                                    turn_lock_dir = +1 if Lm > Rm else -1
                                turn_lock_until = now + TURN_LOCK_SEC

                            if (turn_lock_dir != 0) and (Fm < TURN_LOCK_FRONT_CLEAR) and (now < turn_lock_until):
                                v_t, w_t = 0.0, float(turn_lock_dir)
                                lam = 1.0
                                note = "TURN_LOCK"
                            else:
                                if (turn_lock_dir != 0) and ((now >= turn_lock_until) or (Fm >= TURN_LOCK_FRONT_CLEAR)):
                                    turn_lock_dir = 0

                                door_bias = clamp(door_bias + 0.03, 0.0, 1.0)

                                # ALWAYS compute avoidance and ALWAYS blend
                                v_avoid, w_avoid, L, F, R, used_door = controller_ftg_door(
                                    ranges=ranges, mid=mid, angle_per_index=angle_per_index,
                                    door_bias=door_bias if (now - t0) > DOOR_BIAS_AFTER_SEC else 0.0
                                )

                                if used_door:
                                    door_commit_until = now + DOOR_COMMIT_SEC
                                    door_commit_w = clamp(w_avoid, -DOOR_COMMIT_MAXW, DOOR_COMMIT_MAXW)

                                alpha = clamp((SLOW_FRONT - Fm) / max(SLOW_FRONT - SAFE_FRONT, 1e-6), 0.0, 1.0)
                                lam = clamp(BLEND_LAMBDA_MIN + (BLEND_LAMBDA_MAX - BLEND_LAMBDA_MIN) * alpha,
                                            BLEND_LAMBDA_MIN, BLEND_LAMBDA_MAX)

                                v_t = (1.0 - lam) * v_goal + lam * v_avoid
                                w_t = (1.0 - lam) * w_goal + lam * w_avoid

                                # hard safety
                                if Fm < SAFE_FRONT:
                                    v_t = 0.0
                                    w_t = +1.0 if Lm > Rm else -1.0
                                    lam = 1.0
                                    note = "HARD_SAFE"

                                # door commit
                                if now < door_commit_until:
                                    w_t = clamp(door_commit_w, -DOOR_COMMIT_MAXW, DOOR_COMMIT_MAXW)
                                    v_t = min(v_t, DOOR_COMMIT_V)
                                    note = "DOOR_COMMIT"

            # smooth and drive
            v_s = (1 - SMOOTH) * v_s + SMOOTH * v_t
            w_s = (1 - SMOOTH) * w_s + SMOOTH * w_t
            drive_diff(v_s, w_s, motors)

            # record
            if recording and (step % SAVE_EVERY_N_STEPS == 0):
                t = time.time() - t0
                img_name = f"{step:08d}.jpg"
                img_path = os.path.join(img_dir, img_name)
                try:
                    q.put_nowait((img_path, capture_image(cam)))
                except Full:
                    pass

                wcsv.writerow([f"{t:.3f}", img_name, f"{x:.6f}", f"{y:.6f}", f"{z_h:.6f}",
                               f"{yaw:.6f}", mode, f"{gx:.6f}", f"{gy:.6f}",
                               f"{v_s:.3f}", f"{w_s:.3f}",
                               f"{Lm:.3f}", f"{Fm:.3f}", f"{Rm:.3f}",
                               f"{lam:.2f}", note])

    finally:
        f.close()
        stop_flag["stop"] = True
        try:
            q.put_nowait(None)
        except Exception:
            pass
        print(f"[DONE] Saved: {traj_dir}")

if __name__ == "__main__":
    main()
