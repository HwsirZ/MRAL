# PointGoalNav

## 加载配置

### Habitat-Lab的配置体系

Habitat-Lab 的配置系统为 Hydra，Hydra 是一个开源的 Python 框架，能够通过组合动态创建层级配置，并通过配置文件和命令行覆盖配置。

Hydra 的配置是在运行时动态拼接而成的，有自定义配置以及命令行覆盖的配置以及基础配置组成。

**关于 默认结构化配置 `default_strucured_configs.py`** 

一方面用于配置校验，确保所有必须字段已设置且类型匹配；另一方面也可以直接作为配置。所有结构化配置会注册到 **ConfigStore** 中（内存中的结构化配置注册表）

**habitat-lab config directory** 

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260130195307581.png" alt="image-20260130195307581" style="zoom:50%;" />

```text
habitat-lab/habitat/config/
|
|_benchmark  # benchmark configs (primary configs to be used in habitat.get_config)
| |_nav
| |_rearrange
|
|_habitat    # habitat configs (habitat config groups options)
| |_dataset
| |_simulator
| |_task
|
|_test       # test configs
```



### 以 pointnav_gibson 为实操

选择一个现有的配置文件，比如使用 gibson 数据集，以及执行 pointnav 任务，那么就选择 Habitat-lab 仓库代码下的 `/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_gibson.yaml` 配置文件。  