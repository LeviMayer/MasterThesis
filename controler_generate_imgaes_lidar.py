#!/usr/bin/env python3
# RPLidar A2 (400 rays, 360deg) intelligent exploration + data recording:
# - Follow-The-Gap (FTG) with bubble inflation
# - Door seeking bias (prefers door-like gaps)
# - Door-commit (push through doors / narrow openings)
# - Turn-lock hysteresis near walls (prevents left-right jitter)
# - Deadband for straight driving
# - Front-only target selection (prevents choosing gaps behind)
# - Coverage / exploration memory (avoids "up-down" oscillations in rooms)
# - Records RGB images + Supervisor pose/yaw to go_stanford/manual_recordings/*
#
# Webots: robot controller must be <extern>
# Keys (Webots 3D view focus):
#   R toggle recording
#   P pause
#   Q quit

import os, sys, time, math, csv, random
from datetime import datetime
from queue import Queue, Full, Empty
from threading import Thread

import numpy as np
from PIL import Image as PILImage

# -------- Webots API path ----------
WEBOTS_HOME = os.getenv("WEBOTS_HOME", "/usr/local/webots")
api_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from controller import Supervisor, Keyboard

# -------- Device names (adjust if needed) ----------
CAMERA_NAME = "camera"
LIDAR_NAME  = "lidar"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_MOTOR_NAME  = "left wheel motor"

USE_SELF_NODE = True
ROBOT_DEF_NAME = "TurtleBot3Burger"  # only if USE_SELF_NODE=False

# -------- Output (go_stanford root) ----------
GO_ROOT = "train/vint_train/data/data_splits/go_stanford"
SAVE_EVERY_N_STEPS = 2
JPEG_QUALITY = 85
QUEUE_MAX = 256

# -------- Diff-drive tuning ----------
MAX_WHEEL_SPEED = 6.28
V_GAIN = 0.90 * MAX_WHEEL_SPEED
W_GAIN = 0.60 * MAX_WHEEL_SPEED
SMOOTH = 0.18

# -------- LiDAR preprocessing ----------
MAX_RANGE_CAP = 8.0
MIN_VALID = 0.05
SMOOTH_WINDOW = 7

# Bubble inflation / safety
BUBBLE_RADIUS_M = 0.15
SAFE_FRONT = 0.20
SLOW_FRONT = 0.50

# -------- FTG + door seeking ----------
DOOR_MIN_DIST = 1.0
DOOR_MIN_WIDTH_DEG = 7.0
DOOR_BIAS_AFTER_SEC = 10.0
DOOR_SCORE_BOOST = 2.5

# Door commit
DOOR_COMMIT_SEC = 1.0
DOOR_COMMIT_MAXW = 0.55
DOOR_COMMIT_V = 0.45
DOOR_FRONT_SPAN_DEG = 110.0  # only consider door gaps within +/- this

# Narrow corridor mode
NARROW_LR_THRESH = 0.45
NARROW_SPEED = 0.35

# Stuck detection / recovery
STUCK_WINDOW_SEC = 2.5
STUCK_DIST_M = 0.08
ESCAPE_TURN_SEC = 1.0

# Steering sign (if it turns wrong way, flip this)
STEER_SIGN = -1

# Turn-lock hysteresis
TURN_LOCK_SEC = 0.7
TURN_LOCK_FRONT_CLEAR = 0.55
TURN_LOCK_EPS_SIDE = 0.06

# Deadband for near-straight
DEADBAND_DEG = 4.0

# -------- Coverage / exploration memory (prevents oscillations in rooms) ----------
CELL_SIZE = 0.35
HEADING_UPDATE_SEC = 1.5
NOVELTY_DIST = 1.0
NOVELTY_WEIGHT = 1.6
FREE_WEIGHT = 1.0
HEADING_BLEND = 0.35
YAW_P_GAIN = 1.4
EXPLORE_FRONT_SPAN_DEG = 170

random.seed(0)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def capture_image(camera) -> PILImage.Image:
    data = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (w, h), data).convert("RGB")

def yaw_from_rotation_field(rot):
    ax, ay, az, angle = rot
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9:
        return 0.0
    ax/=n; ay/=n; az/=n
    if abs(ay) >= max(abs(ax), abs(az)):
        return float(angle * (1.0 if ay >= 0 else -1.0))
    half = 0.5 * angle
    s = math.sin(half)
    qw = math.cos(half)
    qx = ax*s; qy = ay*s; qz = az*s
    num = 2*(qw*qy + qx*qz)
    den = 1 - 2*(qy*qy + qx*qx)
    return float(math.atan2(num, den))

def make_run_folder(go_root: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(go_root, "manual_recordings")
    traj_dir = os.path.join(base, f"rplidar_explore_{run_id}")
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

    # prevent too much turning at speed
    if abs(v_cmd) > 0.1:
        w = clamp(w, -0.9*abs(v), 0.9*abs(v))

    v_l = clamp(v - w, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    v_r = clamp(v + w, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    motors[0].setVelocity(v_r)  # right
    motors[1].setVelocity(v_l)  # left

def moving_average(x: np.ndarray, win: int):
    if win <= 1:
        return x
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
    fw = max(3, n // 30)
    front = r[mid - fw: mid + fw]
    left_half = r[mid+fw: mid + n//2] if (mid + n//2) < n else r[mid+fw:]
    right_half = r[mid - n//2: mid-fw] if (mid - n//2) >= 0 else r[:mid-fw]
    L = float(left_half.min()) if left_half.size else float(r.min())
    R = float(right_half.min()) if right_half.size else float(r.min())
    F = float(front.min()) if front.size else float(r.min())
    return L, F, R

def apply_bubble(r: np.ndarray, mid: int, angle_per_index: float):
    n = r.size
    i_min = int(np.argmin(r))
    d_min = float(r[i_min])
    if d_min <= 0.001:
        return r

    theta = math.asin(clamp(BUBBLE_RADIUS_M / d_min, 0.0, 1.0))
    bubble_idx = int(theta / angle_per_index) if angle_per_index > 1e-6 else 5
    bubble_idx = max(3, bubble_idx)

    rr = r.copy()
    lo = max(0, i_min - bubble_idx)
    hi = min(n, i_min + bubble_idx + 1)
    rr[lo:hi] = MIN_VALID
    return rr

def find_gaps(rr: np.ndarray, min_clear: float):
    clear = rr > min_clear
    gaps = []
    n = clear.size
    i = 0
    while i < n:
        if not clear[i]:
            i += 1
            continue
        j = i
        while j < n and clear[j]:
            j += 1
        gaps.append((i, j-1))
        i = j
    return gaps

def angle_to_index(angle_rad: float, mid: int, angle_per_index: float, n: int):
    i = int(round(mid + angle_rad / max(angle_per_index, 1e-6)))
    return int(clamp(i, 0, n - 1))

def choose_target_index_ftg_front(rr: np.ndarray, mid: int, front_span_deg: float, angle_per_index: float):
    n = rr.size
    span = int((math.radians(front_span_deg) / max(angle_per_index, 1e-6)))
    lo = max(0, mid - span)
    hi = min(n, mid + span + 1)

    idx = np.arange(lo, hi, dtype=np.int32)
    ang_pen = np.abs(idx - mid) / float(span + 1)
    score = rr[lo:hi] - 0.6 * ang_pen
    return int(idx[int(np.argmax(score))])

def choose_target_index_door_front(rr: np.ndarray, mid: int, angle_per_index: float, front_span_deg: float):
    n = rr.size
    span = int((math.radians(front_span_deg) / max(angle_per_index, 1e-6)))
    lo = max(0, mid - span)
    hi = min(n, mid + span + 1)

    rr_front = rr[lo:hi]
    gaps = find_gaps(rr_front, DOOR_MIN_DIST)
    if not gaps:
        return None

    best_i = None
    best_score = -1e9
    for (a, b) in gaps:
        width_idx = b - a + 1
        width_deg = width_idx * (angle_per_index * 180.0 / math.pi)
        if width_deg < DOOR_MIN_WIDTH_DEG:
            continue
        seg = rr_front[a:b+1]
        mean_d = float(np.mean(seg))
        cand_local = (a + b) // 2
        cand = lo + cand_local
        turn_pen = abs(cand - mid) / float(n)
        score = (width_deg * mean_d) - 3.0 * turn_pen
        if score > best_score:
            best_score = score
            best_i = cand

    return best_i

def cell_key(x: float, z: float):
    return (int(math.floor(x / CELL_SIZE)), int(math.floor(z / CELL_SIZE)))

def controller_ftg_door(ranges: np.ndarray, mid: int, angle_per_index: float, door_bias: float):
    """
    Returns: v_cmd, w_cmd, L, F, R, used_door(bool)
    """
    r = preprocess_ranges(ranges)
    L, F, R = sectors_min(r, mid)

    if F < SAFE_FRONT:
        v = 0.0
        w = +1.0 if L > R else -1.0
        return v, w, L, F, R, False

    rr = apply_bubble(r, mid, angle_per_index)

    door_i = choose_target_index_door_front(rr, mid, angle_per_index, DOOR_FRONT_SPAN_DEG)
    ftg_i = choose_target_index_ftg_front(rr, mid, front_span_deg=90.0, angle_per_index=angle_per_index)

    used_door = False
    if door_i is not None and door_bias > 0.0:
        if rr[door_i] * DOOR_SCORE_BOOST * door_bias >= rr[ftg_i]:
            target_i = door_i
            used_door = True
        else:
            target_i = ftg_i
    else:
        target_i = ftg_i

    target_angle = (target_i - mid) * angle_per_index

    if abs(target_angle) < math.radians(DEADBAND_DEG):
        w_cmd = 0.0
    else:
        w_cmd = clamp(STEER_SIGN * (target_angle / (math.pi/2)), -1.0, 1.0)

    v_cmd = 0.85
    if F < SLOW_FRONT:
        v_cmd = 0.35
    v_cmd *= clamp(1.0 - 0.5*abs(w_cmd), 0.35, 1.0)

    # narrow mode
    if (L < NARROW_LR_THRESH) and (R < NARROW_LR_THRESH) and (F > SAFE_FRONT):
        v_cmd = min(v_cmd, NARROW_SPEED)
        w_cmd = clamp(w_cmd, -0.55, 0.55)

    return v_cmd, w_cmd, L, F, R, used_door

def pick_exploration_heading(ranges: np.ndarray, mid: int, angle_per_index: float,
                            x: float, z: float, yaw: float, visited_counts: dict):
    """
    Choose a relative heading (rad) that points to open + less-visited area.
    """
    r = preprocess_ranges(ranges)
    n = r.size
    span = math.radians(EXPLORE_FRONT_SPAN_DEG)
    num = 41

    best_score = -1e9
    best_ang = 0.0

    for k in range(num):
        rel_ang = -span/2 + (span * k) / (num - 1)
        idx = angle_to_index(rel_ang, mid, angle_per_index, n)

        free_d = float(r[idx])
        # probe point in world
        wx = x + NOVELTY_DIST * math.cos(yaw + rel_ang)
        wz = z + NOVELTY_DIST * math.sin(yaw + rel_ang)
        seen = visited_counts.get(cell_key(wx, wz), 0)

        score = FREE_WEIGHT * free_d - NOVELTY_WEIGHT * float(seen)
        if score > best_score:
            best_score = score
            best_ang = rel_ang

    return best_ang


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
    rotation_field = node.getField("rotation")

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
    wcsv.writerow(["t", "img", "x", "z", "yaw", "v_cmd", "w_cmd", "L", "F", "R"])

    print("===================================")
    print("RPLidar A2 Explore + Record (FTG + Doors + Coverage Memory)")
    print(f"Lidar: {LIDAR_NAME}, rays={n}, fov_deg={fov*180/math.pi:.1f}")
    print(f"Save dir: {traj_dir}")
    print("Keys (Webots 3D view focus): R rec | P pause | Q quit")
    print("===================================")

    q = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}
    th = Thread(target=writer_worker, args=(q, stop_flag), daemon=True)
    th.start()

    recording = False
    paused = False

    last_progress_check = time.time()
    last_pos = translation_field.getSFVec3f()
    door_bias = 0.0
    escape_until = 0.0

    # turn-lock state
    turn_lock_dir = 0
    turn_lock_until = 0.0

    # door-commit state
    door_commit_until = 0.0
    door_commit_w = 0.0

    # coverage memory
    visited_counts = {}
    last_heading_update = time.time()
    explore_rel_heading = 0.0

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
                motors[0].setVelocity(0.0)
                motors[1].setVelocity(0.0)
                continue

            # lidar
            ranges = np.array(lidar.getRangeImage(), dtype=np.float32)
            if ranges.size != n:
                n = ranges.size
                mid = n // 2
                angle_per_index = fov / max(1, (n - 1))

            # pose + visited update
            x, _, z = translation_field.getSFVec3f()
            yaw = yaw_from_rotation_field(rotation_field.getSFRotation())
            ck = cell_key(x, z)
            visited_counts[ck] = visited_counts.get(ck, 0) + 1

            # update exploration heading periodically
            if (now - last_heading_update) >= HEADING_UPDATE_SEC:
                explore_rel_heading = pick_exploration_heading(
                    ranges=ranges,
                    mid=mid,
                    angle_per_index=angle_per_index,
                    x=x, z=z, yaw=yaw,
                    visited_counts=visited_counts
                )
                last_heading_update = now

            # stuck / progress check
            if (now - last_progress_check) >= STUCK_WINDOW_SEC:
                x0, y0, z0 = last_pos
                x1, y1, z1 = translation_field.getSFVec3f()
                dist = math.hypot(x1 - x0, z1 - z0)

                if dist < STUCK_DIST_M:
                    escape_until = now + ESCAPE_TURN_SEC
                    door_bias = 1.0
                    # push exploration to change direction
                    last_heading_update = 0.0
                else:
                    door_bias = max(0.0, door_bias - 0.25)

                last_pos = (x1, y1, z1)
                last_progress_check = now

            # escape mode
            if now < escape_until:
                r = preprocess_ranges(ranges)
                L, F, R = sectors_min(r, mid)
                escape_dir = +1.0 if L > R else -1.0
                v_t, w_t = 0.12, escape_dir
            else:
                door_bias = clamp(door_bias + 0.02, 0.0, 1.0)

                # compute mins early for turn-lock
                r_dbg = preprocess_ranges(ranges)
                Lm, Fm, Rm = sectors_min(r_dbg, mid)

                # trigger lock only if not already locked
                if (turn_lock_dir == 0) and (Fm < SAFE_FRONT):
                    if abs(Lm - Rm) < TURN_LOCK_EPS_SIDE:
                        turn_lock_dir = random.choice([+1, -1])
                    else:
                        turn_lock_dir = +1 if Lm > Rm else -1
                    turn_lock_until = now + TURN_LOCK_SEC

                # if locked & still close -> keep turning
                if (turn_lock_dir != 0) and (Fm < TURN_LOCK_FRONT_CLEAR) and (now < turn_lock_until):
                    v_t, w_t = 0.0, float(turn_lock_dir)
                else:
                    if (turn_lock_dir != 0) and ((now >= turn_lock_until) or (Fm >= TURN_LOCK_FRONT_CLEAR)):
                        turn_lock_dir = 0

                    v_t, w_t, L, F, R, used_door = controller_ftg_door(
                        ranges=ranges,
                        mid=mid,
                        angle_per_index=angle_per_index,
                        door_bias=door_bias if (now - t0) > DOOR_BIAS_AFTER_SEC else 0.0
                    )

                    # Door commit
                    if used_door:
                        door_commit_until = now + DOOR_COMMIT_SEC
                        door_commit_w = clamp(w_t, -DOOR_COMMIT_MAXW, DOOR_COMMIT_MAXW)

            # apply door-commit (unless escaping)
            if now < door_commit_until and now >= escape_until:
                w_t = clamp(door_commit_w, -DOOR_COMMIT_MAXW, DOOR_COMMIT_MAXW)
                v_t = min(v_t, DOOR_COMMIT_V)

            # blend exploration heading into steering (unless escaping/turn-lock)
            if now >= escape_until and turn_lock_dir == 0:
                w_goal = clamp(STEER_SIGN * (YAW_P_GAIN * explore_rel_heading / (math.pi/2)), -1.0, 1.0)
                w_t = clamp((1.0 - HEADING_BLEND) * w_t + HEADING_BLEND * w_goal, -1.0, 1.0)
                v_t = max(v_t, 0.40)

            # smooth
            v_s = (1 - SMOOTH) * v_s + SMOOTH * v_t
            w_s = (1 - SMOOTH) * w_s + SMOOTH * w_t

            drive_diff(v_s, w_s, motors)

            # record
            if recording and (step % SAVE_EVERY_N_STEPS == 0):
                t = time.time() - t0
                img_name = f"{step:08d}.jpg"
                img_path = os.path.join(img_dir, img_name)

                pil = capture_image(cam)
                try:
                    q.put_nowait((img_path, pil))
                except Full:
                    pass

                r_dbg2 = preprocess_ranges(ranges)
                Lm2, Fm2, Rm2 = sectors_min(r_dbg2, mid)

                wcsv.writerow([f"{t:.3f}", img_name, f"{x:.6f}", f"{z:.6f}", f"{yaw:.6f}",
                               f"{v_s:.3f}", f"{w_s:.3f}", f"{Lm2:.3f}", f"{Fm2:.3f}", f"{Rm2:.3f}"])

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
