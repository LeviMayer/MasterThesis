#!/usr/bin/env python3
# ============================================================
# RPLidar Mapping + A* Goals + Display + Recording
# FIXES INCLUDED:
#   1) Map mirrored -> LIDAR_ANGLE_SIGN switch
#   2) Robot drives away / 180° flipped -> YAW_OFFSET = pi
#   3) Wobble near straight-to-goal -> deadband + turn-in-place
#   4) Robot moves even without goal -> Explore (gap-follow)
#
# Webots:
#   controller: <extern>
#   Add to robot children:
#     Display { name "display" width 256 height 256 }
#
# Console goals:
#   type:  x y   (meters in world ground plane)
# ============================================================

import os, sys, time, math, csv, threading, heapq
import numpy as np
from PIL import Image as PILImage

# -------- Webots API path ----------
WEBOTS_HOME = os.getenv("WEBOTS_HOME", "/usr/local/webots")
api_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from controller import Supervisor, Keyboard, Display

# ---------------- Device names ----------------
CAMERA_NAME = "camera"
LIDAR_NAME = "lidar"
DISPLAY_NAME = "display"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_MOTOR_NAME = "left wheel motor"

# ---------------- Map config ----------------
WORLD_SIZE = 13.0
RES = 0.05
MAP_N = int(WORLD_SIZE / RES)

MAX_RANGE = 8.0
MIN_VALID = 0.05

# Log-odds update + clamp
OCC_INC = 0.70
FREE_DEC = 0.40
L_MIN = -6.0
L_MAX = +6.0
OCC_THRESH = 0.65

# Inflate obstacles for planning (safety margin)
INFLATE_M = 0.20
INFLATE_CELLS = max(1, int(INFLATE_M / RES))

# ---------------- LiDAR geometry / sign ----------------
# You said: map mirrored -> keep -1 (most common fix).
# If after this it becomes mirrored the other way, set +1.
LIDAR_ANGLE_SIGN = -1

# For many Webots 360° lidars, "front" is at n//2.
FRONT_INDEX_OFFSET = 0  # tune +/- 10..50 if needed

# ---------------- Motion config ----------------
MAX_WHEEL = 6.28

# Base wheel speeds (rad/s)
V_BASE = 3.2
W_BASE = 2.4

SAFE_STOP = 0.35
SOFT_STOP = 0.60

# If steering direction feels inverted, flip this
STEER_SIGN = -1

# IMPORTANT: Your robot is 180° flipped relative to yaw computation
# -> this fixes "drives away from goal"
YAW_OFFSET = math.pi  # set 0.0 if it becomes wrong

WALL_FOLLOW_SIDE = "left"  # or "right"

# ---------------- Path following stabilization ----------------
HEADING_DEADBAND_DEG = 4.0
TURN_IN_PLACE_RAD = 0.60   # ~35 degrees
W_MAX_CMD = 0.9

# ---------------- Explore behavior (no goal) ----------------
EXPLORE_FRONT_SPAN_DEG = 120.0
EXPLORE_DEADBAND_DEG = 5.0

# ---------------- Display / Recording ----------------
DISPLAY_EVERY = 10
SAVE_EVERY = 3
GO_ROOT = "train/vint_train/data/data_splits/go_stanford"


# ================= Utility =================
def clamp(x, a, b):
    return max(a, min(b, x))

def wrap_pi(a):
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ================= Pose / Sensors =================
def yaw_from_node_xy(node) -> float:
    """
    Yaw in x-y ground plane.
    NOTE: YAW_OFFSET fixes "robot forward axis" mismatch (180° flipped).
    """
    m = node.getOrientation()  # 3x3 row-major
    fx, fy = m[0], m[3]        # forward projected to XY
    yaw = math.atan2(fy, fx)
    return wrap_pi(yaw + YAW_OFFSET)

def capture_image(camera) -> PILImage.Image:
    data = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (w, h), data).convert("RGB")

def paint_square(img, i, j, color, r=2):
    h, w = img.shape[0], img.shape[1]
    i0 = max(0, i - r); i1 = min(w, i + r + 1)
    j0 = max(0, j - r); j1 = min(h, j + r + 1)
    img[j0:j1, i0:i1] = color


# ================= Occupancy Grid =================
class OccGrid:
    def __init__(self, start_x=0.0, start_y=0.0):
        self.n = MAP_N
        self.res = RES
        # center map around start
        self.ox = float(start_x - WORLD_SIZE / 2.0)
        self.oy = float(start_y - WORLD_SIZE / 2.0)

        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)  # [j,i]
        self.seen = np.zeros((self.n, self.n), dtype=bool)
        self.updated = False

    def world_to_ij(self, x, y):
        i = int((x - self.ox) / self.res)
        j = int((y - self.oy) / self.res)
        return i, j

    def ij_to_world(self, i, j):
        x = self.ox + (i + 0.5) * self.res
        y = self.oy + (j + 0.5) * self.res
        return x, y

    def in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def prob(self):
        lo = np.clip(self.logodds, L_MIN, L_MAX)
        return 1.0 / (1.0 + np.exp(-lo))

    def update_ray(self, x, y, ang_world, dist):
        dist = float(min(dist, MAX_RANGE))
        steps = int(dist / self.res)

        # free along ray
        for k in range(steps):
            px = x + k * self.res * math.cos(ang_world)
            py = y + k * self.res * math.sin(ang_world)
            i, j = self.world_to_ij(px, py)
            if self.in_bounds(i, j):
                self.logodds[j, i] = clamp(self.logodds[j, i] - FREE_DEC, L_MIN, L_MAX)
                self.seen[j, i] = True

        # occupied at end (if hit)
        if dist < MAX_RANGE * 0.999:
            px = x + dist * math.cos(ang_world)
            py = y + dist * math.sin(ang_world)
            i, j = self.world_to_ij(px, py)
            if self.in_bounds(i, j):
                self.logodds[j, i] = clamp(self.logodds[j, i] + OCC_INC, L_MIN, L_MAX)
                self.seen[j, i] = True

        self.updated = True

def inflate_occ(occ: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return occ.copy()
    h, w = occ.shape
    out = occ.copy()
    ys, xs = np.where(occ)
    for y, x in zip(ys, xs):
        x0 = max(0, x - r); x1 = min(w, x + r + 1)
        y0 = max(0, y - r); y1 = min(h, y + r + 1)
        out[y0:y1, x0:x1] = True
    return out


# ================= A* =================
def astar(blocked: np.ndarray, start, goal):
    h, w = blocked.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if blocked[sy, sx] or blocked[gy, gx]:
        return None

    def heur(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = []
    heapq.heappush(pq, (heur((sx, sy), (gx, gy)), 0, (sx, sy)))
    came = {}
    gscore = {(sx, sy): 0}

    while pq:
        _, g, cur = heapq.heappop(pq)
        if cur == (gx, gy):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        x, y = cur
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if blocked[ny, nx]:
                continue
            ng = g + 1
            if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                f = ng + heur((nx, ny), (gx, gy))
                heapq.heappush(pq, (f, ng, (nx, ny)))
    return None


# ================= Path following (stable) =================
def follow_path(robot_xy, yaw, path_world, lookahead=0.45):
    if not path_world or len(path_world) < 2:
        return None

    x, y = robot_xy

    # closest index on path
    dists = [math.hypot(px-x, py-y) for (px, py) in path_world]
    i0 = int(np.argmin(dists))

    # pick lookahead target
    target = path_world[-1]
    acc = 0.0
    for k in range(i0, len(path_world) - 1):
        x1, y1 = path_world[k]
        x2, y2 = path_world[k+1]
        seg = math.hypot(x2-x1, y2-y1)
        acc += seg
        if acc >= lookahead:
            target = (x2, y2)
            break

    tx, ty = target
    ang = math.atan2(ty - y, tx - x)
    err = wrap_pi(ang - yaw)

    # deadband -> no jitter near straight
    if abs(err) < math.radians(HEADING_DEADBAND_DEG):
        err = 0.0

    # turn-in-place if large error
    if abs(err) > TURN_IN_PLACE_RAD:
        v_cmd = 0.0
        w_cmd = clamp(err / (math.pi/2), -W_MAX_CMD, W_MAX_CMD) * STEER_SIGN
        return v_cmd, w_cmd

    w_cmd = clamp(err / (math.pi/2), -W_MAX_CMD, W_MAX_CMD) * STEER_SIGN
    v_cmd = V_BASE * clamp(1.0 - 0.5*abs(w_cmd), 0.25, 1.0)
    return v_cmd, w_cmd


# ================= Explore (gap-follow) =================
def explore_cmd(ranges: np.ndarray, front_idx: int, fov: float):
    n = len(ranges)
    dtheta = fov / float(n)

    span = int((math.radians(EXPLORE_FRONT_SPAN_DEG) / max(dtheta, 1e-6)) / 2.0)
    idxs = (np.arange(front_idx - span, front_idx + span + 1) % n).astype(int)

    rr = ranges[idxs].astype(np.float32)

    rel = np.arange(-span, span+1, dtype=np.float32)
    ang_pen = np.abs(rel) / max(1.0, float(span))
    score = rr - 0.6 * ang_pen

    best_k = int(np.argmax(score))
    best_rel = rel[best_k]
    target_angle = best_rel * dtheta

    if abs(target_angle) < math.radians(EXPLORE_DEADBAND_DEG):
        w_cmd = 0.0
    else:
        w_cmd = clamp(target_angle / (math.pi/2), -1.0, 1.0) * STEER_SIGN

    front_span = max(8, n // 30)
    front_ids = (np.arange(front_idx - front_span, front_idx + front_span + 1) % n).astype(int)
    fmin = float(np.min(ranges[front_ids]))

    v = V_BASE
    if fmin < SOFT_STOP:
        v *= 0.35
    v *= clamp(1.0 - 0.5*abs(w_cmd), 0.25, 1.0)

    return v, w_cmd, fmin


# ================= Display render =================
def draw_map_on_display(display, gmap: OccGrid, robot_xy=None, goal_xy=None, path_ij=None):
    p = gmap.prob()
    occ = p > OCC_THRESH

    img = np.zeros((MAP_N, MAP_N, 3), dtype=np.uint8)
    img[:, :, :] = 120                          # unknown
    img[gmap.seen & (~occ)] = (240, 240, 240)    # free
    img[occ] = (0, 0, 0)                         # occupied

    # path in blue
    if path_ij is not None:
        for (pi, pj) in path_ij[::2]:
            if 0 <= pi < MAP_N and 0 <= pj < MAP_N:
                img[pj, pi] = (40, 120, 255)

    # goal in green
    if goal_xy is not None:
        gi, gj = gmap.world_to_ij(goal_xy[0], goal_xy[1])
        if gmap.in_bounds(gi, gj):
            paint_square(img, gi, gj, (0, 255, 0), r=2)

    # robot in red
    if robot_xy is not None:
        ri, rj = gmap.world_to_ij(robot_xy[0], robot_xy[1])
        if gmap.in_bounds(ri, rj):
            paint_square(img, ri, rj, (255, 0, 0), r=2)

    # y+ up on screen
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


# ================= Main =================
def main():
    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())
    robot.step(ts)

    node = robot.getSelf()
    trans_field = node.getField("translation")

    cam = robot.getDevice(CAMERA_NAME)
    cam.enable(ts)

    lidar = robot.getDevice(LIDAR_NAME)
    lidar.enable(ts)
    try:
        lidar.enablePointCloud(False)
    except Exception:
        pass

    motors = [robot.getDevice(RIGHT_MOTOR_NAME), robot.getDevice(LEFT_MOTOR_NAME)]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    kb = Keyboard()
    kb.enable(ts)

    try:
        display = robot.getDevice(DISPLAY_NAME)
    except Exception:
        display = None
        print('[WARN] No display device. Add: Display { name "display" width 256 height 256 }')

    x0, y0, _ = trans_field.getSFVec3f()
    gmap = OccGrid(start_x=x0, start_y=y0)
    print(f"[MAP] origin=({gmap.ox:.2f},{gmap.oy:.2f}) centered on start=({x0:.2f},{y0:.2f})")

    run = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(GO_ROOT, "manual_recordings", f"slam_like_{run}")
    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)

    traj_f = open(os.path.join(outdir, "traj.csv"), "w", newline="")
    writer = csv.writer(traj_f)
    writer.writerow(["t", "img", "x", "y", "yaw"])

    goal_lock = threading.Lock()
    goal_xy = {"goal": None}

    def console_thread():
        print("Console goal input: type `x y` (meters, ground). Example: 1.2 -0.6")
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    continue
                xs, ys = line.split()
                gx, gy = float(xs), float(ys)
                with goal_lock:
                    goal_xy["goal"] = (gx, gy)
                print("[GOAL SET]", (gx, gy))
            except Exception:
                print("Invalid input. Use: x y")

    threading.Thread(target=console_thread, daemon=True).start()

    # LiDAR geometry
    fov = 2.0 * math.pi
    front_idx = None

    # path state
    path_ij = None
    path_world = None
    last_plan = 0.0
    PLAN_EVERY_SEC = 0.7

    step = 0
    t0 = time.time()

    print("===================================")
    print("Mapping + A* goals + Explore fallback + Display + Recording")
    print(f"LIDAR_ANGLE_SIGN={LIDAR_ANGLE_SIGN} (mirror fix)")
    print(f"YAW_OFFSET={YAW_OFFSET} (180° fix)")
    print("Press Q in Webots window to quit.")
    print("===================================")

    try:
        while robot.step(ts) != -1:
            step += 1
            now = time.time()

            # quit
            key = kb.getKey()
            if key in (ord('Q'), ord('q')):
                break

            x, y, _ = trans_field.getSFVec3f()
            yaw = yaw_from_node_xy(node)

            ranges = np.array(lidar.getRangeImage(), dtype=np.float32)
            ranges = np.where(np.isfinite(ranges), ranges, MAX_RANGE)
            ranges = np.clip(ranges, MIN_VALID, MAX_RANGE)

            n = len(ranges)
            dtheta = fov / float(n)

            if front_idx is None:
                front_idx = (n // 2 + FRONT_INDEX_OFFSET) % n
                print("[INFO] front_idx:", front_idx, "n:", n)

            # -------- Mapping (WORLD-FIXED, not mirrored) --------
            # Correct world angle uses yaw + signed sensor angle
            for i, r in enumerate(ranges):
                ray_ang = LIDAR_ANGLE_SIGN * (i - front_idx) * dtheta
                ang_world = yaw + ray_ang
                gmap.update_ray(x, y, ang_world, float(r))

            # goal read
            with goal_lock:
                g = goal_xy["goal"]

            # build blocked grid for planning
            p = gmap.prob()
            occ = p > OCC_THRESH
            blocked = inflate_occ(occ, INFLATE_CELLS)

            # plan periodically if goal exists
            if g is not None and (now - last_plan) > PLAN_EVERY_SEC:
                sx, sy = gmap.world_to_ij(x, y)
                gx, gy = gmap.world_to_ij(g[0], g[1])

                if gmap.in_bounds(sx, sy) and gmap.in_bounds(gx, gy):
                    # snap goal if blocked
                    if blocked[gy, gx]:
                        found = None
                        for rr in range(1, 18):
                            for dx in range(-rr, rr+1):
                                for dy in range(-rr, rr+1):
                                    nx, ny = gx + dx, gy + dy
                                    if 0 <= nx < MAP_N and 0 <= ny < MAP_N and not blocked[ny, nx]:
                                        found = (nx, ny)
                                        break
                                if found: break
                            if found: break
                        if found is not None:
                            gx, gy = found

                    pi = astar(blocked, (sx, sy), (gx, gy))
                    path_ij = pi
                    if pi is not None and len(pi) >= 2:
                        path_world = [gmap.ij_to_world(ii, jj) for (ii, jj) in pi]
                    else:
                        path_world = None
                else:
                    path_ij = None
                    path_world = None

                last_plan = now

            # safety front min
            front_span = max(8, n // 30)
            idxs = (np.arange(front_idx - front_span, front_idx + front_span + 1) % n).astype(int)
            fmin = float(np.min(ranges[idxs]))

            # -------- Control --------
            v_cmd = 0.0
            w_cmd = 0.0

            if fmin < SAFE_STOP:
                v_cmd = 0.0
                w_cmd = (1.0 if WALL_FOLLOW_SIDE == "left" else -1.0) * 0.9 * STEER_SIGN
            else:
                if g is not None and path_world is not None:
                    out = follow_path((x, y), yaw, path_world, lookahead=0.45)
                    if out is not None:
                        v_cmd, w_cmd = out
                    else:
                        v_cmd, w_cmd, _ = explore_cmd(ranges, front_idx, fov)
                else:
                    v_cmd, w_cmd, _ = explore_cmd(ranges, front_idx, fov)

                if fmin < SOFT_STOP:
                    v_cmd *= 0.35

            # goal reached
            if g is not None:
                dist = math.hypot(g[0] - x, g[1] - y)
                if dist < 0.25:
                    print("[GOAL REACHED]", g)
                    with goal_lock:
                        goal_xy["goal"] = None
                    path_ij = None
                    path_world = None
                    v_cmd, w_cmd = 0.0, 0.0

            # send motor speeds
            vr = clamp(v_cmd + (W_BASE * w_cmd), -MAX_WHEEL, MAX_WHEEL)
            vl = clamp(v_cmd - (W_BASE * w_cmd), -MAX_WHEEL, MAX_WHEEL)
            motors[0].setVelocity(vr)  # right
            motors[1].setVelocity(vl)  # left

            # display
            if display is not None and (step % DISPLAY_EVERY == 0) and gmap.updated:
                draw_map_on_display(display, gmap, robot_xy=(x, y), goal_xy=g, path_ij=path_ij)
                gmap.updated = False

            # record
            if step % SAVE_EVERY == 0:
                img_name = f"{step:06d}.jpg"
                img_path = os.path.join(imgdir, img_name)
                capture_image(cam).save(img_path, "JPEG", quality=85)
                writer.writerow([f"{time.time()-t0:.3f}", img_name, f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}"])

    finally:
        traj_f.close()
        print("[DONE] Saved:", outdir)


if __name__ == "__main__":
    main()
