
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
# Importing essential libraries

import pickle



UPLOAD_FOLDER = './abc/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/images',
            static_folder='./static',
            template_folder='./templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

inception_chest = load_model('./inceptionchest_0.h5')
Pneumonia = load_model('./Pneumonia_0.h5')
# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = Pneumonia.predict(data)
    return predicted
    #
    # image = cv2.imread('./abc/images/upload_chest1.jpg') # read file
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    # image = cv2.resize(image,(224,224))
    # image = np.array(image) / 255
    # image = np.expand_dims(image, axis=0)
    # inception_pred = inception_chest.predict(image)
    # probability = inception_pred[0]
    # return probability


@app.route('/upload1', methods=['POST'])
def upload1():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('index2.html', prediction=my_prediction)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/index1.html', methods=['GET'])
def index1():
    # Main page
    return render_template('index1.html')

@app.route('/index2.html', methods=['GET'])
def index2():
    # Main page
    return render_template('index2.html')

@app.route('/upload11', methods=['POST','GET'])
def upload11_file():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, 'uploadpneumonia.jpg')
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            print(result)
            if(result>0.15):
                label= indices[1]
                accuracy= result

                # text=accuracy+label
                # print(accuracy+label)
            else:
                label= indices[0]
                accuracy= 100-result

                # text=accuracy+label
                # print(accuracy+label)

            return render_template('index1.html',inception_chest_pred=accuracy[0][0],data=label)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Pneumonia"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if  request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if 1==1:
            # filename = secure_filename(file.filename)
            print(1)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest1.jpg'))


    image = cv2.imread('./abc/images/upload_chest1.jpg') # read file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    inception_pred = inception_chest.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
        inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')
    else:
        inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    print(inception_chest_pred)

    return inception_chest_pred


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            print(1)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))


    image = cv2.imread('./abc/images/upload_chest.jpg') # read file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    inception_pred = inception_chest.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
        inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')
    else:
        inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    print(inception_chest_pred)

    return inception_chest_pred


if __name__ == '__main__':
    app.secret_key = ".."
    app.run()
