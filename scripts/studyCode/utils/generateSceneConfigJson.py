import os, json, glob

ROOT = "/data2/wukaitong/gibson/data/scene_datasets/gibson"
OUT  = os.path.join(ROOT, "gibson.scene_dataset_config.json")

glbs = sorted(glob.glob(os.path.join(ROOT, "*.glb")))
if not glbs:
    raise SystemExit(f"[ERROR] No .glb found in {ROOT}")

scenes = []
for glb in glbs:
    base = os.path.splitext(os.path.basename(glb))[0]  # Adrian.glb -> Adrian
    item = {"id": base, "scene_file": os.path.basename(glb)}
    nav = os.path.join(ROOT, base + ".navmesh")
    if os.path.exists(nav):
        item["navmesh"] = os.path.basename(nav)
    scenes.append(item)

cfg = {"dataset": {"name": "gibson", "type": "GibsonDataset-v1", "scenes": scenes}}
os.makedirs(ROOT, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(cfg, f, indent=2)

print("[OK] wrote:", OUT)
print("[OK] scenes:", len(scenes))
print("[OK] example:", scenes[0])


"""
python -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.episodes=1 \
  habitat.dataset.data_path="/data2/wukaitong/gibson/data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz" \
  habitat.simulator.scene_dataset="/data2/wukaitong/gibson/data/scene_datasets/gibson/gibson.scene_dataset_config.json"

"""
