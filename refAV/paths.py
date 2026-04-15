import os
from pathlib import Path


def _default_av2_data_dir() -> Path:
    env_path = os.environ.get("REFAV_AV2_DATA_DIR") or os.environ.get("AV2_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser()

    scratch_candidate = Path.home() / "scratch" / "argoverse_data" / "sensor"
    if scratch_candidate.exists():
        return scratch_candidate

    return Path.home() / Path("data/datasets/sensor")


# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = _default_av2_data_dir()
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('scenario_mining_downloads')

# Required to run evaluation on nuPrompt/nuScenes dataset, ignore otherwise
NUPROMPT_DATA_DIR = Path('/data/nuscenes/nuprompt_v1.0')
NUSCENES_DIR = Path('/data/nuscenes/v1.0-trainval')
NUSCENES_AV2_DATA_DIR = Path('/data/nuscenes/av2_format')

# input directories, do not change
EXPERIMENTS = Path('run/experiment_configs/experiments.yml')
PROMPT_DIR = Path('run/llm_prompting')

# output directories, do not change
SM_DATA_DIR = Path('output/sm_dataset')
SM_PRED_DIR = Path('output/sm_predictions')
LLM_PRED_DIR = Path('output/llm_code_predictions')
TRACKER_PRED_DIR = Path('output/tracker_predictions')
GLOBAL_CACHE_PATH = Path('output/cache')
