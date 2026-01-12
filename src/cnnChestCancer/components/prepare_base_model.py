import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnChestCancer.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config : PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.param_image_size,
            weights=self.config.param_weights,
            include_top=self.config.param_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.base_model)  
        return self.base_model 
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)