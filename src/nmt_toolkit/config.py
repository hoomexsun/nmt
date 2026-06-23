from pathlib import Path
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = PROJECT_ROOT / "conf"


# -------------------------------------------------------- CONFIG LOADING


def _read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs():
    models_cfg = _read_yaml(CONF_DIR / "models.yaml")
    directions_cfg = _read_yaml(CONF_DIR / "directions.yaml")
    jobs_cfg = _read_yaml(CONF_DIR / "jobs.yaml")
    runtime_cfg = _read_yaml(CONF_DIR / "runtime.yaml")
    return models_cfg, directions_cfg, jobs_cfg, runtime_cfg


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


# -------------------------------------------------------- REGISTRY BUILDERS


def build_job_configs():
    models_cfg, directions_cfg, jobs_cfg, runtime_cfg = load_all_configs()

    model_registry = models_cfg["models"]
    direction_registry = directions_cfg["directions"]
    jobs = jobs_cfg["jobs"]
    runtime = runtime_cfg["runtime"]

    merged = []
    for job in jobs:
        model_family = job["model_family"]
        direction_key = job["direction"]

        model_info = model_registry[model_family]
        direction_info = direction_registry[direction_key]
        pair_info = direction_info["pairs"][model_family]

        # Start from global training config
        global_training = dict(runtime_cfg["training"])
        model_training_override = model_info.get("training") or {}
        # Apply per-model overrides on top of global
        merged_training = {**global_training, **model_training_override}

        cfg = {
            "job_name": f"{direction_key}_{model_family}",
            "direction": direction_key,
            "direction_label": direction_info["label"],
            "model_family": model_family,
            "src_lang": pair_info["src_lang"],
            "tgt_lang": pair_info["tgt_lang"],
            "reverse": pair_info.get("reverse", False),
            "tsv_file": resolve_path(direction_info["corpus_file"]),
            "base_exp_dir": resolve_path(runtime["base_exp_dir"]),
            "allow_online": runtime.get("allow_online", False),
            "transformers_offline": runtime.get("transformers_offline", False),
            "hf_repo_id": model_info["repo_id"],
            "hf_model_path": resolve_path(model_info["local_path"]),
            "folder_prefix": model_info["folder_prefix"],
            "model_type": model_info["model_type"],
            "training": merged_training,
            "inference": runtime_cfg["inference"],
            "gui": runtime_cfg["gui"],
            "evaluation": runtime_cfg["evaluation"],
        }

        merged.append(cfg)

    return merged


def build_direction_registry():
    models_cfg, directions_cfg, _, _ = load_all_configs()
    model_registry = models_cfg["models"]
    direction_registry = directions_cfg["directions"]

    items = []
    for direction_key, direction_info in direction_registry.items():
        item = {
            "direction": direction_key,
            "label": direction_info["label"],
            "corpus_file": resolve_path(direction_info["corpus_file"]),
            "pairs": {},
        }
        for model_family, pair in direction_info["pairs"].items():
            item["pairs"][model_family] = {
                **pair,
                "model_family": model_family,
                "hf_repo_id": model_registry[model_family]["repo_id"],
                "hf_model_path": resolve_path(
                    model_registry[model_family]["local_path"]
                ),
                "folder_prefix": model_registry[model_family]["folder_prefix"],
                "model_type": model_registry[model_family]["model_type"],
            }
        items.append(item)
    return items


def maybe_enable_offline_mode(cfg: dict):
    if cfg.get("transformers_offline", False):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
