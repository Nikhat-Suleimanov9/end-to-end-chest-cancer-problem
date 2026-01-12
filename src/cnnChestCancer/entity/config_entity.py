import os
from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    param_image_size: list
    param_include_top: bool
    param_weights: str


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    train_data: Path
    valid_data: Path
    test_data: Path
    param_freeze_n: int
    param_epochs_phase_1: int
    param_epochs_phase_2: int
    param_learning_rate_phase_1: float
    param_learning_rate_phase_2: float
    param_batch_size: int
    param_is_augmentation: bool
    param_do_offline_augm: bool
    param_target_size_augm: int
    param_image_size: list
    param_reduce_lr: list
    param_classes: int    