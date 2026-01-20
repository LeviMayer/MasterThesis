#!/usr/bin/env python3
# ============================================================
# RANDOM GOAL NAV (x,y in [-12,-1]) + BUG2/WALL-FOLLOW USING LIDAR
# + SIMPLE OCCUPANCY MAP (DISPLAY) + CAMERA LOGGING (go_stanford)
#
# Webots:
#   - Robot controller: <extern>
#   - Add to robot children:
#       Display { name "display" width 512 height 512 }
#
# Keys (Webots 3D view focus):
#   R toggle recording
#   P pause
#   Q quit
# ============================================================

import os, sys, time, math, csv, random
from datetime import datetime
import numpy as np
from PIL import Image as PILImage

WEBOTS_HOME = os.getenv("WEBOTS_HOME", "/usr/local/webots")
api_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from controller import Supervisor, Keyboard, Display

# ---------------- Devices ----------------
CAMERA_NAME = "camera"
LIDAR_NAME = "lidar"
DISPLAY_NAME = "display"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_MOTOR_NAME = "left wheel motor"

USE_SELF_NODE = True
ROBOT_DEF_NAME = "TurtleBot3Burger"

# ---------------- Goals ----------------
GOAL_X_MIN, GOAL_X_MAX = -12.0, -1.0
GOAL_Y_MIN, GOAL_Y_MAX = -12.0, -1.0
GOAL_REACH_DIST = 0.35

# ---------------- Motion ----------------
MAX_WHEEL = 6.28
V_CRUISE = 3.0
V_SLOW   = 1.3
W_TURN   = 2.2

SMOOTH_V = 0.18
SMOOTH_W = 0.10
W_RATE_LIMIT = 0.07  # limit normalized w change per step

STEER_SIGN = -1
YAW_OFFSET = math.pi  # try 0.0 if wrong
HEADING_DEADBAND_DEG = 5.0

# ---------------- LiDAR safety ----------------
MAX_RANGE_CAP = 8.0
MIN_VALID = 0.05

SAFE_STOP = 0.28
SOFT_STOP = 0.70

# Bug2 leave conditions
LEAVE_MARGIN = 0.30          # must be at least this much closer than hit distance
LEAVE_HEADING_DEG = 18.0     # need roughly facing goal to leave wall-follow
LEAVE_FRONT_CLEAR = 0.80     # front clearance to leave

# stuck recovery
STUCK_WINDOW_SEC = 2.5
STUCK_DIST_M = 0.08
ESCAPE_SEC = 0.8

# ---------------- Mapping ----------------
WORLD_SIZE_M = 13.0
RES_M = 0.05
MAP_N = int(WORLD_SIZE_M / RES_M)

OCC_INC = 0.80
FREE_DEC = 0.22      # smaller => less flicker/noise
L_MIN, L_MAX = -6.0, 6.0
OCC_THRESH = 0.65

MAP_UPDATE_EVERY_STEPS = 2
RAY_STRIDE = 2
DISPLAY_EVERY_STEPS = 6

# map update gating (reduce smear)
POSE_MIN_MOVE = 0.03          # meters
POSE_MIN_YAW  = math.radians(2.5)

# lidar smoothing
RANGE_SMOOTH_WIN = 7

# ---------------- Recording ----------------
GO_ROOT = "train/vint_train/data/data_splits/go_stanford"
SAVE_EVERY_STEPS = 3
JPEG_QUALITY = 85

PRINT_GOAL_EVERY_SEC = 1.0

random.seed(0)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def capture_image(camera) -> PILImage.Image:
    data = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (w, h), data).convert("RGB")

def make_run_folder(go_root: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(go_root, "manual_recordings")
    traj_dir = os.path.join(base, f"rand_goals_bug2_{run_id}")
    img_dir = os.path.join(traj_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    traj_csv = os.path.join(traj_dir, "traj.csv")
    return traj_dir, img_dir, traj_csv

def yaw_from_node_xy(node) -> float:
    m = node.getOrientation()
    fx, fy = m[0], m[3]          # local X axis projected to world XY
    yaw = math.atan2(fy, fx)
    return wrap_pi(yaw + YAW_OFFSET)

def moving_average_1d(x: np.ndarray, win: int):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, k, mode="valid").astype(np.float32)

def normalize_ranges(r):
    r = np.array(r, dtype=np.float32)
    r = np.where(np.isfinite(r), r, MAX_RANGE_CAP)
    r = np.clip(r, MIN_VALID, MAX_RANGE_CAP)
    r = moving_average_1d(r, RANGE_SMOOTH_WIN)
    return r

def sector_min(ranges, idxs):
    return float(np.min(ranges[idxs])) if len(idxs) else float(np.min(ranges))


# ============================================================
# Occupancy grid (log-odds)
# ============================================================
class OccGrid:
    def __init__(self, start_x, start_y):
        self.n = MAP_N
        self.res = RES_M
        self.ox = float(start_x - WORLD_SIZE_M / 2.0)
        self.oy = float(start_y - WORLD_SIZE_M / 2.0)
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)  # [j,i]
        self.seen = np.zeros((self.n, self.n), dtype=np.bool_)

    def in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def world_to_ij(self, x, y):
        i = int((x - self.ox) / self.res)
        j = int((y - self.oy) / self.res)
        return i, j

    def prob(self):
        lo = np.clip(self.logodds, L_MIN, L_MAX)
        return 1.0 / (1.0 + np.exp(-lo))

    def update_ray(self, x, y, ang, dist):
        dist = float(min(dist, MAX_RANGE_CAP))
        if dist < MIN_VALID:
            return
        steps = int(dist / self.res)
        if steps <= 0:
            return

        cs = math.cos(ang)
        sn = math.sin(ang)

        # free
        for k in range(steps):
            px = x + (k * self.res) * cs
            py = y + (k * self.res) * sn
            i, j = self.world_to_ij(px, py)
            if self.in_bounds(i, j):
                self.logodds[j, i] = clamp(self.logodds[j, i] - FREE_DEC, L_MIN, L_MAX)
                self.seen[j, i] = True

        # occ at endpoint if hit
        if dist < MAX_RANGE_CAP * 0.999:
            px = x + dist * cs
            py = y + dist * sn
            i, j = self.world_to_ij(px, py)
            if self.in_bounds(i, j):
                self.logodds[j, i] = clamp(self.logodds[j, i] + OCC_INC, L_MIN, L_MAX)
                self.seen[j, i] = True


def paint_square(img, i, j, color, r=2):
    h, w = img.shape[0], img.shape[1]
    i0 = max(0, i - r); i1 = min(w, i + r + 1)
    j0 = max(0, j - r); j1 = min(h, j + r + 1)
    img[j0:j1, i0:i1] = color

def draw_map(display: Display, gmap: OccGrid, robot_xy, goal_xy):
    p = gmap.prob()
    seen = gmap.seen
    occ = (p > OCC_THRESH) & seen
    free = (p < (1.0 - OCC_THRESH)) & seen   # ✅ FIXED

    img = np.zeros((MAP_N, MAP_N, 3), dtype=np.uint8)
    img[:, :, :] = 120
    img[free] = (240, 240, 240)
    img[occ] = (0, 0, 0)

    gi, gj = gmap.world_to_ij(goal_xy[0], goal_xy[1])
    if 0 <= gi < MAP_N and 0 <= gj < MAP_N:
        paint_square(img, gi, gj, (0, 255, 0), r=2)

    ri, rj = gmap.world_to_ij(robot_xy[0], robot_xy[1])
    if 0 <= ri < MAP_N and 0 <= rj < MAP_N:
        paint_square(img, ri, rj, (255, 0, 0), r=2)

    img = np.flipud(img)

    pil = PILImage.fromarray(img).resize((display.getWidth(), display.getHeight()), PILImage.NEAREST)
    arr = np.array(pil, dtype=np.uint8)

    h, w = arr.shape[:2]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = arr[..., 2]
    bgra[..., 1] = arr[..., 1]
    bgra[..., 2] = arr[..., 0]
    bgra[..., 3] = 255

    img_ref = display.imageNew(bgra.tobytes(), Display.BGRA, w, h)
    display.imagePaste(img_ref, 0, 0, False)
    display.imageDelete(img_ref)


# ============================================================
# BUG2 controller
# ============================================================
def compute_control_bug2(ranges, front_idx, robot_xy, yaw, goal_xy,
                         wall_mode, hit_goal_dist):
    """
    wall_mode: None | "left" | "right"
    hit_goal_dist: distance to goal when wall-follow started (for leave condition)
    Returns: (v_norm, w_norm, wall_mode, hit_goal_dist)
    """
    n = len(ranges)
    fw = max(6, n // 40)

    front_ids = (np.arange(front_idx - fw, front_idx + fw + 1) % n).astype(int)
    left_ids  = (np.arange(front_idx, front_idx + n//4) % n).astype(int)
    right_ids = (np.arange(front_idx - n//4, front_idx) % n).astype(int)

    F = sector_min(ranges, front_ids)
    L = sector_min(ranges, left_ids)
    R = sector_min(ranges, right_ids)

    gx, gy = goal_xy
    x, y = robot_xy
    goal_dist = math.hypot(gx - x, gy - y)

    if goal_dist < GOAL_REACH_DIST:
        return 0.0, 0.0, None, None

    goal_ang = math.atan2(gy - y, gx - x)
    err = wrap_pi(goal_ang - yaw)

    # --- enter wall-follow if obstacle ahead ---
    if wall_mode is None and F < SAFE_STOP:
        wall_mode = "left" if (L > R) else "right"
        hit_goal_dist = goal_dist
        w = (+1.0 if wall_mode == "left" else -1.0)
        return 0.0, STEER_SIGN * w, wall_mode, hit_goal_dist

    # --- wall-follow mode ---
    if wall_mode is not None:
        desired = 0.50
        if wall_mode == "left":
            side_ids = (np.arange(front_idx + n//8, front_idx + n//4) % n).astype(int)
            side = sector_min(ranges, side_ids)
            e = clamp(desired - side, -0.7, 0.7)
            w = clamp(-1.0 * e, -0.65, 0.65)
        else:
            side_ids = (np.arange(front_idx - n//4, front_idx - n//8) % n).astype(int)
            side = sector_min(ranges, side_ids)
            e = clamp(desired - side, -0.7, 0.7)
            w = clamp(+1.0 * e, -0.65, 0.65)

        # BUG2 leave condition:
        # only leave if we are meaningfully closer than hit point AND heading to goal is reasonable AND front is clear
        if hit_goal_dist is None:
            hit_goal_dist = goal_dist

        if (goal_dist < (hit_goal_dist - LEAVE_MARGIN)
            and abs(err) < math.radians(LEAVE_HEADING_DEG)
            and F > LEAVE_FRONT_CLEAR):
            wall_mode = None
            hit_goal_dist = None
            # proceed to go-to-goal immediately

        v = 0.60 if F > SOFT_STOP else 0.30
        return v, STEER_SIGN * w, wall_mode, hit_goal_dist

    # --- go-to-goal mode ---
    if abs(err) < math.radians(HEADING_DEADBAND_DEG):
        w = 0.0
    else:
        w = clamp(err / (math.pi/2), -0.85, 0.85)

    v = 0.80 if F > SOFT_STOP else 0.45
    v *= clamp(1.0 - 0.45*abs(w), 0.30, 1.0)

    if abs(err) > math.radians(55.0):
        v = 0.15
        w = clamp(err / (math.pi/2), -1.0, 1.0)

    return v, STEER_SIGN * w, None, None


# ============================================================
# Wheel smoothing + w rate limiting
# ============================================================
def rate_limit(prev, target, max_delta):
    d = target - prev
    if d > max_delta:
        return prev + max_delta
    if d < -max_delta:
        return prev - max_delta
    return target

def drive_wheels(motors, v_norm, w_norm, v_prev, w_prev):
    v_target = V_SLOW + (V_CRUISE - V_SLOW) * clamp(v_norm, 0.0, 1.0)

    # rate limit on normalized w
    w_prev_norm = clamp(w_prev / max(W_TURN, 1e-6), -1.0, 1.0)
    w_target_norm = clamp(w_norm, -1.0, 1.0)
    w_rl_norm = rate_limit(w_prev_norm, w_target_norm, W_RATE_LIMIT)
    w_target = W_TURN * w_rl_norm

    v_s = (1 - SMOOTH_V) * v_prev + SMOOTH_V * v_target
    w_s = (1 - SMOOTH_W) * w_prev + SMOOTH_W * w_target

    vr = clamp(v_s + w_s, -MAX_WHEEL, MAX_WHEEL)
    vl = clamp(v_s - w_s, -MAX_WHEEL, MAX_WHEEL)

    motors[0].setVelocity(vr)
    motors[1].setVelocity(vl)

    return v_s, w_s


# ============================================================
# Main
# ============================================================
def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    node = robot.getSelf() if USE_SELF_NODE else robot.getFromDef(ROBOT_DEF_NAME)
    if node is None:
        raise RuntimeError(f"Robot DEF '{ROBOT_DEF_NAME}' not found.")
    trans_field = node.getField("translation")

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

    display = None
    try:
        display = robot.getDevice(DISPLAY_NAME)
    except Exception:
        print('Add Display: Display { name "display" width 512 height 512 }')

    x0, y0, _ = trans_field.getSFVec3f()
    gmap = OccGrid(x0, y0)

    traj_dir, img_dir, traj_csv = make_run_folder(GO_ROOT)
    f = open(traj_csv, "w", newline="")
    wcsv = csv.writer(f)
    wcsv.writerow(["t", "img", "x", "y", "yaw", "goal_x", "goal_y", "wall_mode"])

    n = int(lidar.getHorizontalResolution()) if hasattr(lidar, "getHorizontalResolution") else 400
    fov = float(lidar.getFov()) if hasattr(lidar, "getFov") else (2*math.pi)
    front_idx = n // 2

    print("===================================")
    print("RANDOM GOALS + BUG2/WALLFOLLOW + LIDAR MAP + DISPLAY + LOGGING")
    print(f"Goals: x in [{GOAL_X_MIN},{GOAL_X_MAX}], y in [{GOAL_Y_MIN},{GOAL_Y_MAX}]")
    print(f"Save dir: {traj_dir}")
    print("Keys: R rec | P pause | Q quit")
    print("If drives away from goal: set YAW_OFFSET=0.0 or math.pi. If turns mirrored: flip STEER_SIGN.")
    print("===================================")

    def sample_goal():
        return (random.uniform(GOAL_X_MIN, GOAL_X_MAX),
                random.uniform(GOAL_Y_MIN, GOAL_Y_MAX))

    recording = True
    paused = False
    goal_idx = 0
    goal_xy = sample_goal()
    print(f"[GOAL {goal_idx}] current goal: {goal_xy}")

    wall_mode = None
    hit_goal_dist = None

    v_prev, w_prev = 0.0, 0.0
    step = 0
    t0 = time.time()

    last_goal_print = time.time()

    # map gating
    last_map_pose = (x0, y0, yaw_from_node_xy(node))
    # stuck detection
    last_progress_t = time.time()
    last_progress_pos = (x0, y0)
    escape_until = 0.0

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

            # pose (x,y are ground)
            x, y, _ = trans_field.getSFVec3f()
            yaw = yaw_from_node_xy(node)

            # periodic goal print
            if (now - last_goal_print) >= PRINT_GOAL_EVERY_SEC:
                dist = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
                print(f"[GOAL {goal_idx}] goal={goal_xy} dist={dist:.2f} wall={wall_mode}")
                last_goal_print = now

            # new goal when reached
            if math.hypot(goal_xy[0] - x, goal_xy[1] - y) < GOAL_REACH_DIST:
                goal_idx += 1
                goal_xy = sample_goal()
                wall_mode = None
                hit_goal_dist = None
                print(f"[GOAL {goal_idx}] new goal: {goal_xy}")

            # lidar
            ranges = normalize_ranges(lidar.getRangeImage())
            if ranges.size != n:
                n = ranges.size
                front_idx = n // 2

            # stuck detection
            if (now - last_progress_t) >= STUCK_WINDOW_SEC:
                dx = x - last_progress_pos[0]
                dy = y - last_progress_pos[1]
                d = math.hypot(dx, dy)
                if d < STUCK_DIST_M:
                    escape_until = now + ESCAPE_SEC
                last_progress_pos = (x, y)
                last_progress_t = now

            # escape if stuck (short reverse + turn)
            if now < escape_until:
                v_norm, w_norm = 0.2, 1.0
                v_prev, w_prev = drive_wheels(motors, v_norm, w_norm, v_prev, w_prev)
            else:
                # control
                v_norm, w_norm, wall_mode, hit_goal_dist = compute_control_bug2(
                    ranges=ranges,
                    front_idx=front_idx,
                    robot_xy=(x, y),
                    yaw=yaw,
                    goal_xy=goal_xy,
                    wall_mode=wall_mode,
                    hit_goal_dist=hit_goal_dist
                )
                v_prev, w_prev = drive_wheels(motors, v_norm, w_norm, v_prev, w_prev)

            # mapping update gating (reduce smear)
            if step % MAP_UPDATE_EVERY_STEPS == 0:
                lx, ly, lyaw = last_map_pose
                if (math.hypot(x - lx, y - ly) > POSE_MIN_MOVE) or (abs(wrap_pi(yaw - lyaw)) > POSE_MIN_YAW):
                    dtheta = fov / float(n)
                    for i in range(0, n, RAY_STRIDE):
                        r = float(ranges[i])
                        ang = yaw + (i - front_idx) * dtheta
                        gmap.update_ray(x, y, ang, r)
                    last_map_pose = (x, y, yaw)

            # display: ALWAYS draw periodically (so you see something immediately)
            if display is not None and (step % DISPLAY_EVERY_STEPS == 0):
                draw_map(display, gmap, robot_xy=(x, y), goal_xy=goal_xy)

            # record
            if recording and (step % SAVE_EVERY_STEPS == 0):
                t = now - t0
                img_name = f"{step:08d}.jpg"
                img_path = os.path.join(img_dir, img_name)
                capture_image(cam).save(img_path, "JPEG", quality=JPEG_QUALITY)
                wcsv.writerow([f"{t:.3f}", img_name, f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}",
                               f"{goal_xy[0]:.6f}", f"{goal_xy[1]:.6f}", str(wall_mode)])

    finally:
        f.close()
        print(f"[DONE] Saved: {traj_dir}")


if __name__ == "__main__":
    main()
