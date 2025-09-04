import subprocess

WANDB_PROJ = "SPRINT-AIDO"
CONFIG = "configs/saprot_agg_config.yaml"
EPOCHS = 20

# Tasks: davis, bindingdb, biosnap, biosnap_prot, biosnap_mol, merged
TASKS = ["davis", "bindingdb", "biosnap", "biosnap_prot", "biosnap_mol", "merged"]
SIZE = "large"   # Only large models
REPS = [0, 1, 2, 3, 4]

for task in TASKS:
    for r in REPS:
        exp_id = f"{task.upper()}_AIDO_{SIZE}_R{r}"
        print(f"=== Running {exp_id} ===")

        cmd = [
            "ultrafast-train",
            "--exp-id", exp_id,
            "--task", task,
            "--config", CONFIG,
            "--target-featurizer", "AIDO_P2ST16B",
            "--prot-proj", "agg",
            "--model-size", SIZE,
            "--replicate", str(r),
            "--epochs", str(EPOCHS),
            "--wandb-proj", WANDB_PROJ
        ]

        subprocess.run(cmd, check=True)
