from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired


#image processing 
import cv2
from PIL import Image
import numpy as np

import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Classify Image")

@app.route('/', methods=['GET',"POST"])
# @app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        
        # get image path
        _image = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        image = cv2.imread(_image)
        img_arr = Image.fromarray(image, 'RGB')
        res_img = img_arr.resize((50,50))
        # Make image loadadble into model
        input_image = np.expand_dims(res_img,axis=0)
        # Load model and pass the image
        model = tf.keras.models.load_model(os.path.dirname(__file__)+"/model.h5")
        print("MM",os.path.dirname(__file__)+"/model.h5")
        
        result = model.predict(input_image)
        # 
        label = str(np.argmax(result))
        # And Return Predictions
        if label == "0":
            return "Cat"
        elif label == "1":
            return "Dog"
        elif label == "2":
            return "Monkey"
        elif label == "3":
            return "Parrot"
        elif label == "4":
            return "Elephant"
        elif label == "5":
            return "Bear"
        else:
            return "Error"
       
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)