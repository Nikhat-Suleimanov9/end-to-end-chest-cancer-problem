import tensorflow as tf
import cv2
from tqdm import tqdm
import albumentations as A
from pathlib import Path
from cnnChestCancer.constants import *
from cnnChestCancer.utils.common import read_yaml, create_directories
import os
from dataclasses import dataclass
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from cnnChestCancer.entity.config_entity import TrainingConfig
import shutil

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.base_model = tf.keras.models.load_model(
            self.config.base_model_path
        )
        return self.base_model
    
    def build_full_model(self):
        """
        Attach classifier head on top of self.base_model and set self.model.
        Head architecture mirrors your earlier design.
        """
        b = self.base_model
        x = tf.keras.layers.MaxPooling2D((2,2))(b.output)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.4)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)
        x = tf.keras.layers.Dense(216, activation='relu')(x)
        prediction = tf.keras.layers.Dense(units=self.config.param_classes, activation='softmax')(x)

        full_model = tf.keras.models.Model(inputs=b.input, outputs=prediction)
        self.model = full_model
        return full_model

    def augment_image(self,image):
        augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.35),
            A.GaussianBlur(p=0.3),
            A.ElasticTransform(p=0.25),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.3),
            # Histogram Equalization (CLAHE) (20% chance)
            A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.25),

        ])
        augmented = augmentation(image=image)
        return augmented['image']
    def balance_classes_offline(self, train_data, output_dir):
        target_size = self.config.param_target_size_augm
        
        # Copy original data first
        if not os.path.exists(output_dir):
            shutil.copytree(train_data, output_dir)

        for class_name in os.listdir(output_dir):
            class_path = os.path.join(output_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)
            current_count = len(images)
            print(f"Class '{class_name}': {current_count} -> {target_size} samples")

            pbar = tqdm(total=target_size-current_count)
            while len(images) < target_size:
                # Randomly pick an existing image
                img_name = np.random.choice(images)
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply transformations
                aug_img = self.augment_image(img)

                # Save augmented image
                new_name = f"aug_{len(images)}.jpg"
                save_path = os.path.join(class_path, new_name)
                cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                
                print(f"Saved augmented image: {save_path}")

                images.append(new_name)
                pbar.update(1)
            pbar.close()
        print()
        print("All classes balanced to target size!")




    def train_valid_test_generators(self):
        """
        Create three separate generators for train, validation, and test datasets.
        Each dataset should be in its own folder.
        """
        if self.config.param_do_offline_augm:
            print("Applying offline augmentation to training data...")
            train_dir = self.config.augmented_train_data
            self.balance_classes_offline(self.config.train_data,self.config.augmented_train_data )
        else:
            train_dir = self.config.train_data    
            
        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.param_image_size[:-1],
            batch_size=self.config.param_batch_size,
            interpolation="bilinear"
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.valid_data,  # separate validation folder
            shuffle=False,
            **dataflow_kwargs
        )

        # Test generator
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,  # separate test folder
            shuffle=False,
            **dataflow_kwargs
        )

        # Train generator
        if self.config.param_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=10,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.2,
                zoom_range=0.15,
                horizontal_flip=True,
                vertical_flip=True,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_dir,  # separate train folder
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
    def freeze_all_layers(self):
        for layer in self.base_model.layers:
            layer.trainable = False


    def unfreeze_last_n_layers(self, n):
        for layer in self.base_model.layers[:-n]:
            layer.trainable = False
        for layer in self.base_model.layers[-n:]:
            layer.trainable = True


    def compile_model(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_phase_1(self):

        print("Phase 1: Freezing all layers")
        self.freeze_all_layers()
        self.compile_model(self.config.param_learning_rate_phase_1)

        history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.param_epochs_phase_1,
            verbose = 1
        )

        return history
    def train_phase_2(self):

        print(f"Phase 2: Unfreezing last {self.config.param_freeze_n} layers")

        self.unfreeze_last_n_layers(self.config.param_freeze_n)
        self.compile_model(self.config.param_learning_rate_phase_2)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(**self.config.param_reduce_lr)
        ]

        history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.param_epochs_phase_2,
            callbacks=callbacks,
            verbose = 1
        )

        return history
    def train(self):


        # Load model created in PrepareBaseModel
        self.get_base_model()
        self.build_full_model()
        # Phase 1
        history_1 = self.train_phase_1()

        # Phase 2
        history_2 = self.train_phase_2()

        # Save final trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        return history_1, history_2   