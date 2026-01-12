from cnnChestCancer.constants import *
from cnnChestCancer.utils.common import read_yaml, create_directories
from cnnChestCancer.entity.config_entity import DataIngestionConfig
from cnnChestCancer.entity.config_entity import PrepareBaseModelConfig
from cnnChestCancer.entity.config_entity import TrainingConfig
from cnnChestCancer.entity.config_entity import EvaluationConfig
import os

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH , params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL= config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            param_image_size = self.params.IMAGE_SIZE,
            param_include_top= self.params.INCLUDE_TOP,
            param_weights= self.params.WEIGHTS
        )

        return prepare_base_model_config      

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        train_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest_Cancer", "train")
        valid_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest_Cancer", "valid")
        test_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest_Cancer", "test")
        augmented_train_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest_Cancer_augmented", "train")


        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.base_model_path),
            train_data =Path(train_data),
            valid_data = Path(valid_data),
            test_data = Path(test_data),
            augmented_train_data = Path(augmented_train_data),
            param_freeze_n=params.FREEZE_N,
            param_epochs_phase_1=params.EPOCHS_PHASE_1,
            param_epochs_phase_2=params.EPOCHS_PHASE_2,
            param_learning_rate_phase_1=params.LEARNING_RATE_PHASE_1,
            param_learning_rate_phase_2=params.LEARNING_RATE_PHASE_2,
            param_batch_size=params.BATCH_SIZE,
            param_is_augmentation=params.AUGMENTATION,
            param_do_offline_augm = params.DO_OFFLINE_AUGM,
            param_target_size_augm = params.TARGET_SIZE_AUGM,
            param_image_size=params.IMAGE_SIZE,
            param_reduce_lr= params.CALLBACKS.REDUCE_LR,
            param_classes= params.CLASSES
        )

        return training_config   

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model = self.config.training.trained_model_path,
            valid_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest_Cancer", "valid"),
            mlflow_uri = 'https://dagshub.com/Nikhat-Suleimanov9/end-to-end-chest-cancer-problem.mlflow',
            all_params = self.params,
            param_image_size = self.params.IMAGE_SIZE,
            param_batch_size = self.params.BATCH_SIZE
        )
        return eval_config     