import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        self.class_names = { 0: "Adenocarcinoma", 1: "Large Cell Carcinoma", 2: "Normal", 3: "Squamous Cell Carcinoma" }

    
    def predict(self):
        model = load_model(os.path.join("model","trained_model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = preprocess_input(test_image)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)
        predicted_index = int(result[0])
        prediction = self.class_names[predicted_index]

        return [{
        "image": prediction}]