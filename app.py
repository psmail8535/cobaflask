from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask import jsonify
from threading import Lock
import pickle #, scipy
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer	
import sqlite3				                                  
# ~ from nltk.corpus import stopwords
# ~ import nltk

# ~ import tensorflow as tf
# ~ from tensorflow import keras
# ~ import keras

import os, sys

# ~ import re
# ~ import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.tree import DecisionTreeClassifier
# ~ from nltk.corpus import stopwords
# ~ stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


app = Flask(__name__)
api = Api(app)
# ~ nltk.download('stopwords')



categories = [  
		'Gender','Perempuan','Anak','Kemiskinan','Rentan',
		'Disabilitas','Minoritas','Lansia','Kasta/Etnik','Tertinggal',
		'Masyarakat Adat','Buruh','Pembantu RT','Pengungsi','Korban']
		
model_cat = [
		'DTree_Gender.sav', 'DTree_Perempuan.sav', 'DTree_Anak.sav','DTree_Kemiskinan.sav','DTree_Rentan.sav',
		'DTree_Disabilitas.sav', 'DTree_Minoritas.sav', 'DTree_Lansia.sav','DTree_Kasta_Etnik.sav', 'DTree_Tertinggal.sav', 
		'DTree_Masyarakat_Adat.sav','DTree_Buruh.sav', 'DTree_Pembantu_RT.sav', 'DTree_Pengungsi.sav','DTree_Korban.sav'

		 ]		



lst_model = []
idx = 0
for category in categories:
	filename = 'model/dtree_model/%s' % (model_cat[idx])
	loaded_model = pickle.load(open(filename, 'rb'))
	lst_model.append(loaded_model)
	print('Load model: ', filename)
	idx += 1


# ~ def getGesiDataset(dir_data, p_test_size=0.2):
	# ~ dts = load_files(dir_data)

	# ~ print('dts: ',(dts.keys()))
	# ~ print('dts: ',(dts['target']))


	# ~ print('data len: ',len(dts['data']))
	# ~ print('target len: ',len(dts['target']))

	# ~ X_train, X_test, y_train, y_test = train_test_split( dts['data'], dts['target'], 
		# ~ test_size=p_test_size, random_state=1, stratify=dts['target'])
	
	# ~ return X_train, X_test, y_train, y_test


# ~ X_train, X_test, y_train, y_test = getGesiDataset('train')

# ~ inggris = stopwords.words('english')
# ~ indo = stopwords.words('indonesian')
n_features = 2 ** 16
vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=2, stop_words=None
                                 #use_idf=opts.use_idf
                                 )  
                                 
# ~ X_train = vectorizer.fit_transform(X_train)  
# ~ X_test = vectorizer.transform(X_test) 



# ~ filename = 'DecisionTree.sav'
filename = 'SVC.sav'
modelSvc = pickle.load(open(filename, 'rb'))
filename = 'SVC.sav'
modelDtree = pickle.load(open(filename, 'rb'))
# ~ filename = 'modelTF564.h5'
# ~ modelTF = None
# ~ with tf.device('/device:cpu:0'):
# ~ modelTF = keras.models.load_model(filename)

labels = ['Gesi','Non Gesi']


# ~ def createSingleTestDataTF(p_judul):
	# ~ lst_test = [p_judul]
	# ~ single_test = vectorizer.transform(lst_test) 
	# ~ single_test = scipy.sparse.csr_matrix.todense(single_test)
	# ~ return single_test
	
def createSingleTestData(p_judul):
	lst_test = [p_judul]
	single_test = vectorizer.transform(lst_test) 
	return single_test

		 
def createSingleTestDataMultiLabel(p_judul):
	single_test = pd.Series(p_judul)
	return single_test
	
def prediksi(p_judul, p_metode):
	result = ''
	label = ''
	metod = ''
	if p_metode == '1':
		v_single_test = createSingleTestData(p_judul)
		print('Metode: Decision Tree')
		metod = 'Decision Tree'
		result = modelDtree.predict(v_single_test)
		label = labels[result[0]]
	elif p_metode == '2':
		v_single_test = createSingleTestData(p_judul)
		print('Metode: SVD')
		metod = 'SVD'
		result = modelSvc.predict(v_single_test)
		label = labels[result[0]]
	# ~ elif p_metode == '3':		
		# ~ v_single_test2 = createSingleTestDataTF(p_judul)
		# ~ print('Metode: Deep Learning')
		# ~ metod = 'Deep Learning'
		# ~ result2 = modelTF.predict(v_single_test2)
		# ~ label = labels[result2.argmax()]
	elif p_metode == '10':		
		v_single_test = createSingleTestDataMultiLabel(p_judul)
		
		print('Metode: Decision Tree Multi Label')
		metod = 'Decision Tree Multi Label'
		
		kategori_res = []
		str_kat_result = ''
		str_idx = ''
		idx = 0
		for category in categories:
			print('... Processing {}'.format(category))
			result = lst_model[idx].predict_proba(v_single_test)	
			print('Prediksi proba: ',result)
			resultF = result.flatten() 
			if resultF[0] == 0:
				kategori_res.append(category)
				str_kat_result += category+','
				str_idx += str(idx) + '#'
			print('===============================')
			idx += 1
		print('kategori_res: ', kategori_res)
		print('str_kat_result: ', str_kat_result)
		
		label = str_kat_result + str_idx
	print('Prediksi: ', label)		
	return label, metod


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/users', methods=['GET'])
def get_users():
    # ~ users = [ user.json() for user in User.query.all() ] 
    return jsonify({'users': 'test' })

@app.route("/login_json", methods=["POST"])
def login():
	judul = request.json.get('judul')
	print('judul: ', judul)
	
	str_pred, metod = prediksi(judul, '10')
	print('str_pred: ',str_pred)
	print('metod: ',metod)
	return jsonify({'prediksi': str_pred, 'metode': metod})
	

@app.route('/api_prediksi/', methods=['POST'])
def api_prediksi():
	str_pred, metod = prediksi(data, metode)
	print('str_pred: ',str_pred)
	print('metod: ',metod)
	return jsonify({'prediksi': str_pred, 'metode': metod})
	

@app.route('/api/<metode>/<data>', methods=['GET'])
def get_user(metode,data):
	print('data: ', data)
	print('metode: ', metode)
	str_pred, metod = prediksi(data, metode)
	print('str_pred: ',str_pred)
	print('metod: ',metod)
	return jsonify({'prediksi': str_pred, 'metode': metod})
	


if __name__ == '__main__':
    # ~ app.run(debug=True, host='192.168.20.3', port=5500)
    app.run(debug=True, host='0.0.0.0', port=5500)
