
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask import jsonify
from threading import Lock
import pickle 
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer	
import sqlite3		
import os, sys, json

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'the_data.sqlite3')
def db_connect(db_path=DEFAULT_PATH):
    con = sqlite3.connect(db_path)
    return con

app = Flask(__name__)
api = Api(app)

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

n_features = 2 ** 16
vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=2, stop_words=None
                                 #use_idf=opts.use_idf
                                 )  
                          
filename = 'SVC.sav'
modelSvc = pickle.load(open(filename, 'rb'))
filename = 'SVC.sav'
modelDtree = pickle.load(open(filename, 'rb'))

labels = ['Gesi','Non Gesi']

	
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
		str_kat_result2 = ''
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
				str_kat_result2 += category+', '
				str_idx += str(idx) + '#'
			print('===============================')
			idx += 1
		print('kategori_res: ', kategori_res)
		print('str_kat_result: ', str_kat_result)
		
		label = str_kat_result + str_idx
	print('Prediksi: ', label)		
	return label, metod, str_kat_result2


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def index_history():
    return render_template('index_history.html')




@app.route('/json_judul')
def json_judul():
	conn = db_connect()
	# ~ cur = con.cursor() 
	query = "select * from daftar_judul" 
	# ~ cur.execute(query)
	# ~ listdata = cur.fetchall()
	# ~ con.commit()
	
	conn.row_factory = sqlite3.Row
	df = pd.read_sql(query, conn)
	conn.close()
	print('df: ',df)
	
	result = df.to_json(orient="records")
	parsed = json.loads(result)
	# ~ json_data = json.dumps(parsed)  
	
	
	# ~ json_data = df.to_json()
	print('parsed: ',parsed)
	
	return jsonify({'data': parsed})

@app.route('/api/users', methods=['GET'])
def get_users():
    # ~ users = [ user.json() for user in User.query.all() ] 
    return jsonify({'users': 'test' })

@app.route("/login_json", methods=["POST"])
def login():
	judul = request.json.get('judul')
	print('judul: ', judul)
	
	str_pred, metod, str_kat_result = prediksi(judul, '10')
	print('str_pred: ',str_pred)
	print('metod: ',metod)
	
	if len(judul) > 0:
		con = db_connect()
		cur = con.cursor() 
		query = "insert into artikel_pred (judul, prediksi) values('%s','%s')" % (judul, str_kat_result)
		print('query: ', query)
		cur.execute(query)
		con.commit()
		
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
    app.run()
