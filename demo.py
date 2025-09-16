import argparse
from pathlib import Path

import torch
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from NLRSC.solver import quadratic_flow
from NLRSC.utils import feats_sampling

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./NLRSC/demo", help="Input directory containing RS frames")
parser.add_argument("--output", type=str, default="out", help="Output directory for results")
parser.add_argument("--gamma", type=float, default=0.5, help="Readout ratio (scanning time per row)")
parser.add_argument("--tau", type=float, default=0.0, help="Target timestamp for GS reconstruction (typically 0)")
parser.add_argument("--fconfig", type=str, default="raft_8x2_100k_mixed_368x768", 
                   help="MMFlow config file name (without extension)")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use: cpu | cuda:0")
args = parser.parse_args()

def main():
    # Initialize paths and directories
    input_dir, output_dir = Path(args.input), Path(args.output)
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg")))
    
    # Verify we have enough frames (I_{-1}, I_0, I_1)
  #  assert len(image_paths) >= 3, "Need at least 3 consecutive frames (I_{-1}, I_0, I_1)"
    I_minus1_path, I0_path, I1_path = image_paths[0], image_paths[1], image_paths[2]
    
    # Initialize optical flow model
    MIM_CACHE = Path("~/.cache/mim").expanduser()
    config_file = MIM_CACHE / f"{args.fconfig}.py"
    checkpoint_file = MIM_CACHE / f"{args.fconfig}.pth"
    flow_model = init_model(str(config_file), str(checkpoint_file), device=args.device)
    
    # Compute optical flows F_{0→-1} and F_{0→1}
    F0_minus1 = inference_model(flow_model, [str(I0_path)], [str(I_minus1_path)], [None])[0]["flow"]  # I0→I-1
    F0_1 = inference_model(flow_model, [str(I0_path)], [str(I1_path)], [None])[0]["flow"]            # I0→I1
    
    # Convert to tensors and move to device
    F0_minus1_t = torch.from_numpy(F0_minus1).unsqueeze(0).to(args.device)  # (1,h,w,2)
    F0_1_t = torch.from_numpy(F0_1).unsqueeze(0).to(args.device)            # (1,h,w,2)
    
    # Compute correction field D_corr using Proposition 2
    D_corr = quadratic_flow(F0_minus1_t, F0_1_t, args.gamma, args.tau)     # (1,h,w,2)
    
    # Load and warp the RS image I0
    transform = transforms.Compose([transforms.ToTensor()])
    I0_RS = transform(Image.open(I0_path).convert("RGB")).unsqueeze(0).to(args.device)  # (1,3,h,w)
    I0_GS = feats_sampling(I0_RS, -D_corr)  # Warp using correction field
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(I0_GS, output_dir / "I0_GS.png")
    visualize_flow(F0_minus1, str(output_dir / "F0_minus1.png"))
    visualize_flow(F0_1, str(output_dir / "F0_1.png"))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()