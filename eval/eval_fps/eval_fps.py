# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the script to measure the inference FPS of the models.
"""
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import argparse
import torch
import yaml
import numpy as np

from envs.warp_sim_envs import RenderMode

from envs.neural_environment import NeuralEnvironment
from utils.torch_utils import num_params_torch_model
from utils.python_utils import set_random_seed
from utils import torch_utils
from utils.evaluator import NeuralSimEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--env-name', 
                        default = 'Pendulum',
                        type = str)
    parser.add_argument('--model-path',
                        default = None,
                        type = str)
    parser.add_argument('--env-mode',
                        default = 'neural',
                        type = str,
                        choices = ['neural', 'ground-truth'])
    parser.add_argument('--num-envs', 
                        default = 2048,
                        type = int)
    parser.add_argument('--rollout-horizon',
                        default = 100,
                        type = int)
    parser.add_argument('--seed', 
                        default = 0,
                        type = int)
    
    args = parser.parse_args()

    device = 'cuda:0'

    set_random_seed(args.seed)

    env_cfg = {
        "env_name": args.env_name,
        "num_envs": args.num_envs,
        "render": False,
        "warp_env_cfg": {
            "seed": args.seed
        },
        "default_env_mode": args.env_mode,
    }
    
    # Load neural model and neural_integrator_cfg if model_path exists
    if args.model_path is not None:
        model, robot_name = torch.load(args.model_path, map_location='cuda:0', weights_only=False)
        # print('Number of Model Parameters: ', num_params_torch_model(model))
        model.to(device)
        train_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)), '../'
        ))
        cfg_path = os.path.join(train_dir, 'cfg.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader = yaml.SafeLoader)
        env_cfg["neural_integrator_cfg"] = cfg["env"]["neural_integrator_cfg"]
    else:
        model = None

    neural_env = NeuralEnvironment(
        neural_model = model,
        **env_cfg
    )
    
    if model is not None:
        assert neural_env.robot_name == robot_name, \
            "neural_env.robot_name is not equal to neural_model's robot_name."
        
    evaluator = NeuralSimEvaluator(
        neural_env, 
        None, 
        args.rollout_horizon, 
        device = device
    )

    set_random_seed(args.seed)

    # We use the same number of trajectories as environments for FPS measurement
    # to maximize parallelism
    num_rollouts = args.num_envs

    _, _, _, fps = \
        evaluator.evaluate_action_mode(
            num_traj = num_rollouts,
            trajectory_source = "sampler",
            eval_mode = "rollout",
            env_mode = args.env_mode,
            passive = True,
            measure_fps = True,
            render = False,
            export_video = False
        )

    print(f'FPS: {fps}')
