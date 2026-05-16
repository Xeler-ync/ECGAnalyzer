import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from utils._config import RANDOM_FOREST_PARAMS, RandomForestParams
from utils._train import run_training

DEFAULT_RF_CONFIG_PATH = "./config/rf_lvh_norm.json"


def load_or_create_rf_config(config_path: str) -> RandomForestParams:
    if not os.path.exists(config_path):
        return _create_rf_config(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    default_dict = {
        k: getattr(RANDOM_FOREST_PARAMS, k)
        for k in RandomForestParams.__dataclass_fields__.keys()
    }
    merged = {**default_dict, **config_dict}
    if isinstance(merged["class_weight"], dict):
        merged["class_weight"] = {int(k): v for k, v in merged["class_weight"].items()}
    return RandomForestParams(**merged)


def _create_rf_config(config_path) -> RandomForestParams:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    default_dict = {
        k: getattr(RANDOM_FOREST_PARAMS, k)
        for k in RandomForestParams.__dataclass_fields__.keys()
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_dict, f, indent=4, ensure_ascii=False)
    print(f"Created default RF config file: {config_path}")
    print("Please edit it if needed, then rerun the script.")
    return RandomForestParams(**default_dict)


def main():
    parser = argparse.ArgumentParser(description="Train RF: LVH vs NORM")
    parser.add_argument(
        "--lvh-dir",
        type=str,
        default="./results/LVH_100",
        help="Directory containing LVH data (positive class)",
    )
    parser.add_argument(
        "--norm-dir",
        type=str,
        default="./results/NORM_100",
        help="Directory containing NORM data (negative class)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/model/lvh_norm_rf",
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--pure-norm",
        action="store_true",
        default=True,
        help="If set, NORM samples must have is_pure_norm == 1",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_RF_CONFIG_PATH,
        help=f"Path to JSON config file for RandomForest hyperparameters. Default: {DEFAULT_RF_CONFIG_PATH}",
    )
    args = parser.parse_args()

    # 加载或创建随机森林配置文件
    rf_params = load_or_create_rf_config(args.config)

    # 调用训练函数
    run_training(
        pos_dir=args.lvh_dir,
        neg_dir=args.norm_dir,
        out_dir=args.out_dir,
        pure_neg=args.pure_norm,
        rf_params=rf_params,
    )


if __name__ == "__main__":
    main()
