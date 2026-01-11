import os
import itertools
import stat

# Sweep Parameters
capi_dims = [32, 64, 128]
lambda_lies = [0.1, 0.08, 0.05, 0.01]
gammas = [1.0, 0.5, 0.1]

# Configuration
folder_name = "SWEEP" # 指定資料夾名稱
script_prefix = "SW"  # 指定腳本代號前綴

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), folder_name)
os.makedirs(output_dir, exist_ok=True)

# Template
# Note: "cd ../.." from scripts/SWEEP/ goes to project root
template = """#!/bin/bash
#PBS -P yp87
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=18GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

RUN_TAG="capi_sweep"

module load cuda/12.6.2

source /scratch/yp87/sl5952/CAPI/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -u src/main.py \\
  --dataset stanford_cars --download --model resnet50 --pretrained \\
  --epochs 300 \\
  --capi_dim {capi_dim} \\
  --lambda_lie {lambda_lie} \\
  --gamma {gamma} \\
  --seed 42 >> "{log_file}" 2>&1
"""

submit_lines = []
count = 0

for dim, lie, gamma in itertools.product(capi_dims, lambda_lies, gammas):
    count += 1
    # Create filename SW00X.sh
    filename = f"{script_prefix}{count:03d}.sh"
    filepath = os.path.join(output_dir, filename)
    
    # Log file name with parameters for easy identification
    log_file = f"{script_prefix}{count:03d}.log"

    content = template.format(
        capi_dim=dim,
        lambda_lie=lie,
        gamma=gamma,
        log_file=log_file
    )
    
    with open(filepath, "w") as f:
        f.write(content)
        
    submit_lines.append(f"qsub {filename}")

# Generate submit_all.sh
submit_all_path = os.path.join(output_dir, "submit_all.sh")
with open(submit_all_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("\n".join(submit_lines))
    f.write("\n")

print(f"Generated {count} scripts in {output_dir}")
print(f"Generated submit script: {submit_all_path}")
