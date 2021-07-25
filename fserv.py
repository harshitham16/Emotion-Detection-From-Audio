from flask import Flask, render_template, request, redirect, url_for
from keras.models import model_from_json
from keras.optimizers import Adam
from werkzeug.utils import secure_filename
import librosa
import pandas as pd
import numpy as np
import sys
import os
import re
#from tensorflow import keras
#model = keras.models.load_model('mymodel')
emotionMap = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprise'}
global model
#sys.path.append(os.path.abspath("./mymodel"))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('test.html', value="")



def init():
  json_file = open('mymodel/update.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("mymodel/update.h5")
  opt = Adam(lr=0.001)
  loaded_model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
  app.logger.error('model loaded')
  #print("Loaded Model from disk")
  return loaded_model


model = init()

app.config['UPLOAD_FOLDER'] = "./static"
def melextrfn(name):
    X, sample_rate = librosa.load(name, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    df = pd.DataFrame(columns=['mel_spectrogram'])
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    log_spectrogram = np.mean(db_spec, axis = 0)
    df.loc[0] = [log_spectrogram] 
    df = pd.DataFrame(df['mel_spectrogram'].values.tolist())
    colstest = np.arange(242,262,1)
    dftest = pd.DataFrame(index=np.arange(1), columns=colstest)
    dftest = dftest.fillna(0) 
    df = pd.concat([df,dftest],axis=1)
    x_log = df.iloc[:, 3:]
    x_log=np.array(x_log)
    sout = x_log[:,:,np.newaxis]
    return sout


@app.route('/predictn',methods=['GET', 'POST'])
def predictn():
    audfile = request.files['filename']
    audfile.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(audfile.filename)))
    audpath = str(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(audfile.filename)))
    app.logger.error(audpath)
    #return render_template('test.html', value="okay")
    sample = melextrfn(audpath)
    predictions = model.predict(sample)
    classes = pd.DataFrame(np.argmax(predictions, axis = 1))
    emotepred = str(classes.replace(emotionMap))
    emotepred = re.findall("[a-zA-Z]+", emotepred)
    return render_template('test.html', value=("Emotion detected is "+emotepred[0]))

@app.route("/pagetwo")
def pagetwo():
    return render_template("page2.html")

@app.route("/pagethree")
def pagethree():
    return render_template("page3.html")

if __name__ == '__main__' :
    app.run(debug=True)