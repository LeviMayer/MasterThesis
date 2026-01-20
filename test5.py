#connect navigate to webots

import os, sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv


WEBOTS_HOME = os.getenv('WEBOTS_HOME', '/usr/local/webots')
api_path   = os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python')
if api_path not in sys.path:
    sys.path.insert(0, api_path)

from PIL import Image as PILImage
#from diffusers.schedulers.scheduling_ddpm import DDPMSchedule
from diffusers import DDPMScheduler
#from controller import Robot#, Camera, Lidar, Motor
from controller.robot import Robot
from vint_train.training.train_utils import get_action
from typing import List, Tuple, Dict, Optional
from torchvision import transforms
from torchvision.transforms import functional as TF
from vint_train.models.nomad.nomad import NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.gnm.gnm import GNM
from vint_train.models.nomad.nomad import DenseNetwork
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

# CONSTANTS
MODEL_CONFIG_PATH = "deployment/config/models.yaml"
MAX_V = 1.5  # Max linear velocity
MAX_W = 1.0  # Max angular velocity

# tuning gains
K_v     = 25    # scales forward speed
K_omega = 2.5    # scales how aggressively you turn
WHEEL_BASE = 0.25

goalpicture = PILImage.open('/home/levi/Pictures/Screenshots/Screenshot from 2025-07-17 14-19-47.png')

# GLOBALS
context_queue = []
context_size = None

# Initialize Webots Robot
timestep = 128
robot = Robot()

# Devices fpr Webots
camera = robot.getDevice("camera")
camera.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

motors = [robot.getDevice("right wheel motor"), robot.getDevice("left wheel motor")]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std =[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        pil_img = pil_img.convert("RGB")
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

# Helper functions
def capture_image() -> PILImage:
    """Capture an image from the Webots camera."""
    image_data = camera.getImage()
    width, height = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (width, height), image_data).convert("RGB")

""" def send_waypoint_to_robot(waypoint: np.ndarray):
    ""Send waypoint as motor commands to the robot.""
    left_speed = MAX_V * (1 - waypoint[1])  # Adjust based on steering
    right_speed = MAX_V * (1 + waypoint[1])
    motors[0].setVelocity(right_speed)
    motors[1].setVelocity(left_speed) """



def send_waypoint_to_robot(waypoint: np.ndarray):
    import math

    # 1) unpack
    x, y = waypoint

    # 2) compute v and ω
    v = K_v * waypoint[0]
    ω       = K_omega * math.atan2(y, x)

    # 3) diff-drive inverse kinematics
    v_l = v - (ω * WHEEL_BASE / 2.0)
    v_r = v + (ω * WHEEL_BASE / 2.0)

    # 4) saturate
    v_l = max(-MAX_V, min(MAX_V, v_l))
    v_r = max(-MAX_V, min(MAX_V, v_r))

    # 5) send (check your motor ordering!)
    motors[1].setVelocity(v_l)   # left wheel
    motors[0].setVelocity(v_r)   # right wheel

def main(args):
    global context_size

    # Load configurations
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # Load model
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    steps = 0

    '''#Plot Path
    plt.ion()
    fig, ax = plt.subplots()
    xs, ys = [], []
    path_line, = ax.plot(xs, ys, '-o', markersize=3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Live 2D Trajectory (X vs Z)')
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.show()'''
    #trajectory = []  # list of (x, y)
    positions = []        # Liste von (x, y) = (X, Z)
    collision_flags = [0]  # Liste von 0/1 für Kollision

    print("Starting exploration...")
    while robot.step(timestep) != -1:
        steps += 1
        # Capture image
        obs_img = capture_image()
        if context_size is not None:
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)
        if steps % 10 == 0:
            if len(context_queue) > model_params["context_size"]:
                # Preprocess images
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1).to(device)

                fake_goal  = transform_images(goalpicture, model_params["image_size"], center_crop=False)# = torch.randn((1, 3, *model_params["image_size"])).to(device)
                fake_goal =fake_goal.to(device=device, dtype=obs_images.dtype)
                
                mask = torch.ones(1, device=device, dtype=torch.long)#mask = torch.ones(1).long().to(device)

                # Infer action
                with torch.no_grad():
                    obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                    obs_cond = obs_cond.repeat(args.num_samples, 1) if len(obs_cond.shape) == 2 else obs_cond.repeat(args.num_samples, 1, 1)

                    noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])

                    for k in noise_scheduler.timesteps:
                        noise_pred = model('noise_pred_net', sample=noisy_action, timestep=k, global_cond=obs_cond)
                        noisy_action = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_action).prev_sample

                # Extract action and send to robot
                actions = to_numpy(get_action(noisy_action))
                chosen_waypoint = actions[0][args.waypoint]
                if model_params["normalize"]:
                    chosen_waypoint *= (MAX_V / timestep)
                send_waypoint_to_robot(chosen_waypoint)
                print("Sent waypoint:", chosen_waypoint)
                
                x,y,_ = gps.getValues()
                positions.append((x, y))

                print("X: ", x , "   Y: " , y,  collision_flags[-1])

                i = len(positions) - 1
                if i < 4:
                    collision_flags.append(0)
                else:
                    last5 = positions[i - 4 : i + 1]  # fünf (x,z)-Paare
                    x_vals = [round(p[0], 2) for p in last5]
                    z_vals = [round(p[1], 2) for p in last5]
                    if len(set(x_vals)) == 1 and len(set(z_vals)) == 1:
                        collision_flags.append(1)
                    else:
                        collision_flags.append(0)



                with open("trajectory_with_collisions.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y", "collision"])
                    for (px, pz), coll in zip(positions, collision_flags):
                        writer.writerow([f"{px:.5f}", f"{pz:.5f}", coll])
                
                """ x, y, z = gps.getValues()
                xs.append(x)
                ys.append(y)
                path_line.set_data(xs, ys)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001) """

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNM Diffusion Exploration in Webots")
    parser.add_argument("--model", "-m", default="nomad", type=str, help="Model name")
    parser.add_argument("--waypoint", "-w", default=2, type=int, help="Waypoint index")
    parser.add_argument("--num-samples", "-n", default=8, type=int, help="Number of samples")
    args = parser.parse_args()
    main(args)
