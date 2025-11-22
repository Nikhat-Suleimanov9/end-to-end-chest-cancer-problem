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
        self.model = tf.keras.applications.resnet50.ResNet50(
            input_shape=self.config.param_image_size,
            weights=self.config.param_weights,
            include_top=self.config.param_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)   

    @staticmethod
    def _prepare_full_model(b_model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in b_model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in b_model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in b_model.layers[-freeze_till:]:
                layer.trainable = True
        else:
        # default: all layers trainable
            for layer in b_model.layers:
                layer.trainable = True    


        x = tf.keras.layers.GlobalAveragePooling2D()(b_model.output)
        prediction = tf.keras.layers.Dense(
        units=classes,
        activation="softmax"
        )(x)

        
        full_model = tf.keras.models.Model(
            inputs=b_model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()  


        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            b_model=self.model,
            classes=self.config.param_classes,
            freeze_all=False,
            freeze_till=32,
            learning_rate=self.config.param_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)         