#connect navigate with webots

import time
import torch
import os
import yaml
from pathlib import Path
import math
from controller import Robot, Motor, Lidar, Camera
from PIL import Image
import numpy as np

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.gnm.gnm import GNM
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

#things for the robot
timestep = 64
robot = Robot()

camera = robot.getDevice("camera")
camera.enable(timestep)

# Get the motors and enable velocity control
right_motor = robot.getDevice("right wheel motor")
left_motor = robot.getDevice("left wheel motor")
right_motor.setPosition(float('inf'))
left_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)
left_motor.setVelocity(0.0)

image_counter = 0

while robot.step(timestep) != -1:
    left_speed = 1.5
    right_speed = 1.5

    # Apply computed velocities
    right_motor.setVelocity(right_speed)
    left_motor.setVelocity(left_speed)
    
    image_data = camera.getImage()
    
    # Convert the raw image to an array
    width = camera.getWidth()
    height = camera.getHeight()
    image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))  # RGBA format

    # Convert array to Image object
    image = Image.fromarray(image_array)
     
    if image_counter%10 == 0:
        # Save the image
        image.save("C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/Webot_Pictures/1/camera_image"+str(image_counter)+".png")
        print("Image saved as 'captured_image"+str(image_counter)+".png'")
    image_counter = image_counter+1



# Define the path to the YAML configuration files and model weights
MODEL_WEIGHTS_PATH = "C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/model/visualnav-transformer-main/deployment/model_weights"
MODEL_CONFIG_PATH = "C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/model/visualnav-transformer-main/deployment/config/models.yaml"

# Load the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model configuration from YAML
with open(MODEL_CONFIG_PATH, "r") as f:
    model_paths = yaml.safe_load(f)

# Specify the model to load (assuming "nomad" in this case)
model_name = "nomad"
model_config_path = "C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/model/visualnav-transformer-main/train/config/nomad.yaml"
print(f"Loading model configuration from {model_config_path}")
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

# Load model weights from the checkpoint file
ckpth_path = model_paths[model_name]["ckpt_path"]
if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
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



# Load the model
model = load_model(ckpth_path, model_params, device)
model = model.to(device)
model.eval()

import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((85, 64)),      # Resize to 85x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         # Convert to a tensor with shape (3, 85, 64)
])
image1 = transform(Image.open("C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/Webot_Pictures/1/camera_image10.png").convert("RGB"))
image2 = transform(Image.open("C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/Webot_Pictures/1/camera_image20.png").convert("RGB"))
image3 = transform(Image.open("C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/Webot_Pictures/1/camera_image30.png").convert("RGB"))
image4 = transform(Image.open("C:/Users/levid/Desktop/Uni/Master/SoSe24/ProjektLearningRobots/Webot_Pictures/1/camera_image40.png").convert("RGB"))
image = torch.stack([image1,image2,image3,image4], dim=0).permute(1, 0, 2, 3).reshape(1, 12, 85, 64)

#image = torch.zeros((1, 12,85,64))
goal = torch.zeros((1, 3,85,64))
obsgoal_cond = model('vision_encoder', obs_img=image,goal_img = goal, input_goal_mask= None)
dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
#infere Action
noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
obs_cond = obsgoal_cond
num_samples = 1
with torch.no_grad():
    # encoder vision features
    if len(obs_cond.shape) == 2:
        obs_cond = obs_cond.repeat(num_samples, 1)
    else:
        obs_cond = obs_cond.repeat(num_samples, 1, 1)
    
    # initialize action from Gaussian noise
    noisy_action = torch.randn(
        (num_samples, model_params["len_traj_pred"], 2), device=device)
    naction = noisy_action

    # init scheduler
    noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])

    start_time = time.time()
    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            'noise_pred_net',
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )
        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample
    print("time elapsed:", time.time() - start_time)
from matplotlib import pyplot as plt
plt.scatter(naction[0, :, 0].cpu().numpy(), naction[0, :, 1].cpu().numpy())
plt.show()
print("Model loaded successfully!")
print(model)
