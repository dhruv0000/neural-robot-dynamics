import torch
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.models import ModelMixedInput

def verify():
    # Mock config
    try:
        with open('train/cfg/Cartpole/transformer.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback if running from a different directory
        with open('neural-robot-dynamics/train/cfg/Cartpole/transformer.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

    # Disable normalization for verification
    if 'network' in cfg:
        cfg['network']['normalize_input'] = False
        cfg['network']['normalize_output'] = False

    # Create dummy input sample based on config
    # Ant env usually has these dimensions, but exact values don't matter for init
    # We need to match the keys in cfg['inputs']['low_dim']
    
    # ['states_embedding', 'contact_normals', 'contact_points_1', 'contact_depths', 'joint_acts', 'gravity_dir']
    
    # Assuming batch size 2, sequence length 10
    B, T = 2, 10
    
    input_sample = {}
    # We need to guess dimensions or look at env. 
    # Let's assume some dimensions. 
    # states_embedding: usually state dim. Ant is ~29?
    # contact_normals: 3 * num_contacts?
    # Let's just give it some random dimensions, MLPBase will adapt to input size.
    
    input_sample['states_embedding'] = torch.randn(B, T, 29)
    input_sample['contact_normals'] = torch.randn(B, T, 12) # 4 contacts * 3
    input_sample['contact_points_1'] = torch.randn(B, T, 12)
    input_sample['contact_depths'] = torch.randn(B, T, 4)
    input_sample['joint_acts'] = torch.randn(B, T, 8)
    input_sample['gravity_dir'] = torch.randn(B, T, 3)
    
    print("Instantiating Jamba model...")
    try:
        model = ModelMixedInput(
            input_sample=input_sample,
            output_dim=29, # Output dim matches state dim usually
            input_cfg=cfg['inputs'],
            network_cfg=cfg['network'],
            device='cpu',
            novelty='jamba'
        )
        print("Model instantiated successfully.")
        print(model)
        
        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running forward pass...")
    try:
        input_dict = {k: v.clone() for k, v in input_sample.items()}
        output = model(input_dict)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Failed forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
