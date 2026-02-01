python -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav.yaml \
  habitat_baselines.trainer_name=ppo \
  habitat_baselines.evaluate=False \
  habitat_baselines.num_environments=1 \
  habitat.dataset.split=train \
  habitat.dataset.data_path="/data2/wukaitong/gibson/data/datasets/pointnav/gibson/v1/train/train.json.gz/train.json.gz" \
  habitat.simulator.scene_dataset="/data2/wukaitong/gibson/data/scene_datasets/gibson/gibson.scene_dataset_config.json" \
  habitat_baselines.num_updates=10 \
  habitat_baselines.total_num_steps=-1
