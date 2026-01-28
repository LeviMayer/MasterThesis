from controller import Robot, Keyboard, Camera
import os
import csv
import time

# controller for Webots: drive with WASD and save camera images + labels
# Save in driveWithKeysAndSafeAll.py


TIME_STEP = 32  # ms
MAX_SPEED = 6.28  # adjust to your robot
SAVE_DIR = "dataset"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABELS_PATH = os.path.join(SAVE_DIR, "labels.csv")


def setup_folders():
    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "timestamp", "vx", "vz", "left_speed", "right_speed"])


def main():
    robot = Robot()
    keyboard = Keyboard()
    keyboard.enable(TIME_STEP)

    # Get devices (change names to match your robot)
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)

    setup_folders()
    image_idx = 0

    # Control state
    vx = 0.0  # forward speed
    vz = 0.0  # turn speed

    while robot.step(TIME_STEP) != -1:
        key = keyboard.getKey()
        # reset each step
        vx = 0.0
        vz = 0.0

        # WASD control
        if key == ord('W'):
            vx = 1.0
        elif key == ord('S'):
            vx = -1.0
        if key == ord('A'):
            vz = 1.0
        elif key == ord('D'):
            vz = -1.0

        left_speed = (vx - vz) * MAX_SPEED
        right_speed = (vx + vz) * MAX_SPEED

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # Save image + label every step
        filename = f"img_{image_idx:06d}.png"
        path = os.path.join(IMG_DIR, filename)
        camera.saveImage(path, 100)

        with open(LABELS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, time.time(), vx, vz, left_speed, right_speed])

        image_idx += 1


if __name__ == "__main__":
    main()