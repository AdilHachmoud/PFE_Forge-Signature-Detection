from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
MyModel = tf.keras.models.load_model('the model/my_model.h5')

# Define image height and width
img_height = 256
img_width = 256

# Define the class names
class_names = ['fake', 'real']

@app.route('/')
def home():
    return render_template('templates/home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the form
    img_file = request.files['image']
    
    # Load the image
    img = image.load_img(img_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = MyModel.predict(img_array)
    pred_label = class_names[np.argmax(predictions)]


    return render_template('templates/predict.html', prediction=pred_label)

if __name__ == '__main__':
    app.run(debug=True)
