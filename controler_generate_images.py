#!/usr/bin/env python3
# EXTERN Webots teleop recorder (Supervisor pose) + smooth control + async image saving
#
# In Webots 3D window focus:
#   W = forward (latched briefly -> won't drop during turning)
#   S = backward
#   A = turn left (only while recently pressed)
#   D = turn right
#   SPACE = stop (v,w=0)
#   R = toggle recording
#   P = pause (motors 0)
#   Q = quit

import os
import sys
import time
import math
import csv
from datetime import datetime
from queue import Queue, Full, Empty
from threading import Thread

# ---- Headless safe (ok even with GUI Webots) ----
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

# ---- Webots controller API path ----
WEBOTS_HOME = os.getenv("WEBOTS_HOME", "/usr/local/webots")
api_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from controller import Supervisor, Keyboard
from PIL import Image as PILImage


# ----------------------------
# Robot/device config (like your working setup)
# ----------------------------
CAMERA_NAME = "camera"
RIGHT_MOTOR_NAME = "right wheel motor"
LEFT_MOTOR_NAME = "left wheel motor"

USE_SELF_NODE = True
ROBOT_DEF_NAME = "TurtleBot3Burger"  # only used if USE_SELF_NODE=False

# ----------------------------
# Drive tuning
# ----------------------------
MAX_WHEEL_SPEED = 6.28  # rad/s (Webots typical)
V_GAIN = 0.75 * MAX_WHEEL_SPEED      # stronger forward
W_GAIN = 0.35 * MAX_WHEEL_SPEED      # moderate turn

FWD_CMD = 1.0     # command magnitude (0..1)
TURN_CMD = 1.0    # command magnitude (0..1)

# "Event hold" timeouts:
# - forward latch longer so W doesn't drop when you also press A/D
HOLD_V_SEC = 0.80
HOLD_W_SEC = 0.25

# smoothing (makes it non-jerky)
SMOOTH = 0.5  # 0..1 (higher reacts faster)

# ----------------------------
# Recording config (go_stanford root)
# ----------------------------
GO_ROOT = "train/vint_train/data/data_splits/go_stanford"
SAVE_EVERY_N_STEPS = 10        # IMPORTANT: don't save every step (disk+JPEG stalls)
JPEG_QUALITY = 85             # lower quality = faster
QUEUE_MAX = 128               # image write buffer


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def capture_image(camera) -> PILImage.Image:
    img_data = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    return PILImage.frombytes("RGBA", (w, h), img_data).convert("RGB")


def yaw_from_rotation_field(rotation_sf):
    ax, ay, az, angle = rotation_sf
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return 0.0
    ax /= norm; ay /= norm; az /= norm

    if abs(ay) >= max(abs(ax), abs(az)):
        return float(angle * (1.0 if ay >= 0 else -1.0))

    half = 0.5 * angle
    s = math.sin(half)
    qw = math.cos(half)
    qx = ax * s
    qy = ay * s
    qz = az * s

    num = 2.0 * (qw * qy + qx * qz)
    den = 1.0 - 2.0 * (qy * qy + qx * qx)
    return float(math.atan2(num, den))


def make_run_folder(go_root: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(go_root, "manual_recordings")
    traj_dir = os.path.join(base, f"manual_{run_id}")
    img_dir = os.path.join(traj_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    traj_csv = os.path.join(traj_dir, "traj.csv")
    return traj_dir, img_dir, traj_csv


def drive_diff(v_cmd, w_cmd, motors):
    """
    v_cmd: [-1..1], w_cmd: [-1..1]
    """
    v_cmd = clamp(v_cmd, -1.0, 1.0)
    w_cmd = clamp(w_cmd, -1.0, 1.0)

    v = V_GAIN * v_cmd
    w = W_GAIN * w_cmd

    # Keep curves: allow turning while moving, but don't let turn dominate
    if abs(v_cmd) > 0.05:
        w = clamp(w, -0.75 * abs(v), 0.75 * abs(v))

    v_l = v - w
    v_r = v + w

    v_l = clamp(v_l, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    v_r = clamp(v_r, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    motors[0].setVelocity(v_r)  # right
    motors[1].setVelocity(v_l)  # left


def image_writer_worker(q: Queue, stop_flag: dict):
    """
    Writes PIL images to disk asynchronously, so control loop stays smooth.
    Queue entries: (img_path, pil_image)
    """
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
        except Exception as e:
            # don't crash control loop
            pass
        finally:
            q.task_done()


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # Node fields (Supervisor pose, no GPS)
    if USE_SELF_NODE:
        node = robot.getSelf()
    else:
        node = robot.getFromDef(ROBOT_DEF_NAME)
        if node is None:
            raise RuntimeError(f"Robot DEF '{ROBOT_DEF_NAME}' not found.")

    translation_field = node.getField("translation")
    rotation_field = node.getField("rotation")

    # Devices
    camera = robot.getDevice(CAMERA_NAME)
    camera.enable(timestep)

    motors = [robot.getDevice(RIGHT_MOTOR_NAME), robot.getDevice(LEFT_MOTOR_NAME)]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    kb = Keyboard()
    kb.enable(timestep)

    # Output
    traj_dir, img_dir, traj_csv = make_run_folder(GO_ROOT)
    f = open(traj_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["t", "img", "x", "z", "yaw", "v_cmd", "w_cmd"])

    print("===================================")
    print("EXTERN Teleop + Recording (smooth + async saving)")
    print(f"Save dir: {traj_dir}")
    print("Click Webots 3D view window for keyboard focus!")
    print("W/S forward/back, A/D turn, SPACE stop, R rec, P pause, Q quit")
    print("===================================")

    # async image writer
    q = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}
    th = Thread(target=image_writer_worker, args=(q, stop_flag), daemon=True)
    th.start()

    # timestamps for latching behavior
    last_w = last_s = last_a = last_d = 0.0

    recording = False
    paused = False
    step = 0
    t0 = time.time()

    # smoothed commands
    v_s = 0.0
    w_s = 0.0

    try:
        while robot.step(timestep) != -1:
            step += 1
            now = time.time()

            # Drain keyboard events this step
            key = kb.getKey()
            while key != -1:
                if key in (ord("W"), ord("w")):
                    last_w = now
                elif key in (ord("S"), ord("s")):
                    last_s = now
                elif key in (ord("A"), ord("a")):
                    last_a = now
                elif key in (ord("D"), ord("d")):
                    last_d = now
                elif key == ord(" "):
                    last_w = last_s = last_a = last_d = 0.0
                elif key in (ord("R"), ord("r")):
                    recording = not recording
                    print(f"[INFO] recording = {recording}")
                elif key in (ord("P"), ord("p")):
                    paused = not paused
                    print(f"[INFO] paused = {paused}")
                elif key in (ord("Q"), ord("q")):
                    print("[INFO] quit")
                    return
                key = kb.getKey()

            # Latching logic:
            w_pressed = (now - last_w) < HOLD_V_SEC
            s_pressed = (now - last_s) < HOLD_V_SEC
            a_pressed = (now - last_a) < HOLD_W_SEC
            d_pressed = (now - last_d) < HOLD_W_SEC

            # v target (independent of turn)
            v_t = 0.0
            if w_pressed:
                v_t += FWD_CMD
            if s_pressed:
                v_t -= FWD_CMD
            v_t = clamp(v_t, -1.0, 1.0)

            # w target (independent)
            w_t = 0.0
            if a_pressed:
                w_t += TURN_CMD
            if d_pressed:
                w_t -= TURN_CMD
            w_t = clamp(w_t, -1.0, 1.0)

            # Smooth
            v_s = (1.0 - SMOOTH) * v_s + SMOOTH * v_t
            w_s = (1.0 - SMOOTH) * w_s + SMOOTH * w_t

            if paused:
                motors[0].setVelocity(0.0)
                motors[1].setVelocity(0.0)
                continue

            drive_diff(v_s, w_s, motors)

            # record (async image writing)
            if recording and (step % SAVE_EVERY_N_STEPS == 0):
                t = time.time() - t0
                x, y, z = translation_field.getSFVec3f()
                rot = rotation_field.getSFRotation()
                yaw = yaw_from_rotation_field(rot)

                img_name = f"{step:08d}.jpg"
                img_path = os.path.join(img_dir, img_name)

                # Capture quickly and enqueue for background saving
                pil = capture_image(camera)
                try:
                    q.put_nowait((img_path, pil))
                except Full:
                    # If disk can't keep up, drop frames rather than killing control smoothness
                    pass

                writer.writerow([f"{t:.3f}", img_name, f"{x:.6f}", f"{z:.6f}", f"{yaw:.6f}", f"{v_s:.3f}", f"{w_s:.3f}"])

    finally:
        # clean shutdown
        f.close()
        stop_flag["stop"] = True
        try:
            q.put_nowait(None)
        except Exception:
            pass
        print(f"[DONE] Saved: {traj_dir}")


if __name__ == "__main__":
    main()
