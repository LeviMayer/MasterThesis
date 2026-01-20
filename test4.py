# connect explore with webots
import os
import yaml
import numpy as np
import torch
import torch.nn as nn

from PIL import Image as PILImage

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from controller import Robot, Camera, Lidar, Motor
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
TOPOMAP_IMAGES_DIR = "deployment/topomaps/images"
MODEL_CONFIG_PATH = "deployment/config/models.yaml"

MAX_V = 0.5  # Max linear velocity
MAX_W = 1.0  # Max angular velocity

# GLOBALS
context_queue = []
context_size = None


# Webots setup
timestep = 512
robot = Robot()

# Devices
camera = robot.getDevice("camera")
camera.enable(timestep)

motors = [robot.getDevice("right wheel motor"), robot.getDevice("left wheel motor")]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

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
def capture_image() -> PILImage:
    """Capture an image from the Webots camera."""
    image_data = camera.getImage()
    width, height = camera.getWidth(), camera.getHeight()
    return PILImage.frombytes("RGBA", (width, height), image_data).convert("RGB")


def send_waypoint_to_robot(waypoint: np.ndarray):
    """Send waypoint as motor commands to the robot."""
    left_speed = MAX_V * (1 - waypoint[1])  # Adjust for steering
    right_speed = MAX_V * (1 + waypoint[1])
    motors[0].setVelocity(right_speed)
    motors[1].setVelocity(left_speed)


def main(args):
    global context_size

    # Load model configuration
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # Load model weights
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    # Load topomap
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0]))
    topomap = [PILImage.open(os.path.join(topomap_dir, filename)) for filename in topomap_filenames]

    closest_node = 0
    goal_node = args.goal_node if args.goal_node != -1 else len(topomap) - 1
    reached_goal = False

    # Initialize diffusion scheduler
    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    print("Starting navigation...")
    while robot.step(timestep) != -1:
        # Capture image
        obs_img = capture_image()
        if context_size is not None:
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

        # Navigation logic
        if len(context_queue) > model_params["context_size"]:
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(device)

            goal_image = transform_images(topomap[goal_node], model_params["image_size"], center_crop=False).to(device)

            # Infer actions
            mask = torch.zeros(1).long().to(device)
            obs_cond = model("vision_encoder", obs_img=obs_images, goal_img=goal_image, input_goal_mask=mask)

            noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                noise_pred = model("noise_pred_net", sample=noisy_action, timestep=k, global_cond=obs_cond)
                noisy_action = noise_scheduler.step(noise_pred, timestep=k, sample=noisy_action).prev_sample

            actions = to_numpy(get_action(noisy_action))
            chosen_waypoint = actions[0][args.waypoint]
            if model_params["normalize"]:
                chosen_waypoint *= (MAX_V / timestep)
            send_waypoint_to_robot(chosen_waypoint)

        # Check if goal is reached
        reached_goal = closest_node == goal_node
        if reached_goal:
            print("Goal reached!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Webots-based GNM Diffusion Exploration")
    parser.add_argument("--model", "-m", default="nomad", type=str, help="Model name (default: nomad)")
    parser.add_argument("--waypoint", "-w", default=2, type=int, help="Waypoint index (default: 2)")
    parser.add_argument("--dir", "-d", default="topomap", type=str, help="Path to topomap images")
    parser.add_argument("--goal-node", "-g", default=-1, type=int, help="Goal node index (default: -1 for last node)")
    parser.add_argument("--num-samples", "-n", default=8, type=int, help="Number of action samples (default: 8)")
    args = parser.parse_args()

    main(args)
