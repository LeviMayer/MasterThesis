#!/usr/bin/env python3
# Controller with live trajectory plotting using Supervisor API (no GPS)

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn

# ---- Qt/Matplotlib headless fix ----
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ------------------------------------

# --- ADDITION: Insert Webots controller API path ---
WEBOTS_HOME = os.getenv('WEBOTS_HOME', '/usr/local/webots')
api_path = os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python')
if api_path not in sys.path:
    sys.path.insert(0, api_path)

# --- IMPORT Webots Supervisor ---
from controller import Supervisor
from PIL import Image as PILImage
from diffusers import DDPMScheduler
from vint_train.training.train_utils import get_action
from torchvision import transforms
from torchvision.transforms import functional as TF
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
from typing import List

# CONSTANTS
MODEL_CONFIG_PATH = "deployment/config/models.yaml"
MAX_V = 0.5       # Max linear velocity (m/s)
K_V = 25          # Forward speed gain
K_OMEGA = 2.0     # Steering gain
WHEEL_BASE = 0.25 # Distance between wheels (m)

# Helper functions (transform_images, to_numpy, load_model, send_waypoint_to_robot) as before...


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
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
def capture_image(camera) -> PILImage:
    """Capture an RGB image from the Webots camera device."""
    img_data = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    # Webots returns raw bytes in BGRA order; convert to PIL
    return PILImage.frombytes('RGBA', (width, height), img_data).convert('RGB')



def send_waypoint_to_robot(waypoint, motors):
    """Compute wheel velocities from waypoint and send to motors list [right, left]."""
    import math
    x, y = waypoint
    v = K_V * x
    omega = K_OMEGA * math.atan2(y, x)
    # differential drive inverse kinematics
    v_l = v - (omega * WHEEL_BASE / 2.0)
    v_r = v + (omega * WHEEL_BASE / 2.0)
    # saturate
    v_l = max(-MAX_V, min(MAX_V, v_l))
    v_r = max(-MAX_V, min(MAX_V, v_r))
    # apply
    # motors order: [right_motor, left_motor]
    motors[0].setVelocity(v_r)
    motors[1].setVelocity(v_l)



def main(args):
    # --- INSERTION 1: Use Supervisor instead of Robot ---
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # --- INSERTION 2: Get Supervisor node for trajectory ---
    # In your .wbt set Robot DEF name to MY_TURTLEBOT
    node = robot.getSelf()#robot.getFromDef("TurtleBot3Burger")
    robot.getFromDef("TurtleBot3Burger")
    translation_field = node.getField("translation")

    # Camera setup (unchanged)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Motors setup (unchanged)
    motors = [robot.getDevice("right wheel motor"), robot.getDevice("left wheel motor")]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    # --- INSERTION 3: Live plot setup ---
    #plt.ion()
    fig, ax = plt.subplots()
    xs, zs = [], []
    (line,) = ax.plot([], [], marker="o", linestyle="-", linewidth=1)#ax.plot(xs, zs, marker='o')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Live Robot Trajectory')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal', 'box')
    EPS = 1e-3
    SAVE_EVERY = 10
    #plt.show()

    # Load model config
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    with open(model_paths[args.model]["config_path"], "r") as f:
        model_params = yaml.safe_load(f)
    context_size = model_params.get("context_size", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_paths[args.model]["ckpt_path"], model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # Initialize context queue
    context_queue: List[PILImage.Image] = []

    print("Starting exploration with Supervisor-based trajectory...")
    step_count = 0
    while robot.step(timestep) != -1:
        step_count += 1
        # Capture and enqueue image
        obs_img = capture_image(camera)
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

        # Model inference every timestep or as desired
        if len(context_queue) > context_size:
            # Preprocess images
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)

            # Example: run inference (replace with your own code)
            actions = your_inference_function(model, noise_scheduler, obs_images, args)
            chosen_waypoint = actions[0][args.waypoint]
            send_waypoint_to_robot(chosen_waypoint, motors)

        # Trajectory logging & live plot update
        x, y, z = translation_field.getSFVec3f()
        xs.append(x)
        zs.append(z)
        line.set_data(xs, zs)
        #ax.relim()
        #ax.autoscale_view()
        #plt.draw()
        #plt.pause(0.001)
        #if step_count % 10 == 0:
        #    fig.savefig("/tmp/trajectory.png", dpi=120)
        xmin, xmax = min(xs), max(xs)
        zmin, zmax = min(zs), max(zs)
        dx = max(xmax - xmin, EPS)
        dz = max(zmax - zmin, EPS)
        cx = 0.5 * (xmax + xmin)
        cz = 0.5 * (zmax + zmin)
        r = max(dx, dz) * 0.6  # Rand
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cz - r, cz + r)

        if len(xs) >= 2 and (step_count % SAVE_EVERY == 0):
            try:
                fig.savefig("/tmp/trajectory.png", dpi=120)
            except Exception:
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNM Supervisor Exploration with Live Plot")
    parser.add_argument("--model", "-m", default="vint", type=str)
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    args = parser.parse_args()
    main(args)
