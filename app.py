from flask import Flask, request, Response
from flask_pymongo import PyMongo
import json

from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np

app = Flask(__name__)
MONGO_URL = "mongodb+srv://gHamilton:***REMOVED***@cluster0.j7fth.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
app.config["MONGO_URI"] = MONGO_URL
mongo = PyMongo(app)


# Function to split the dataset
def splitdataset(trainData, testData):
    # Separating the target variable
    x_train = trainData.values[:, 1:]
    x_test = testData.values[:, 1:55]
    y_train = trainData.values[:, 0]
    y_test = testData.values[:, 0]

    return x_train, x_test, y_train, y_test


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    #print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    #print("Accuracy : \n", accuracy_score(y_test, y_pred) * 100)

    #print("Report : \n", classification_report(y_test, y_pred))

    #print("F1 Score: \n", f1_score(y_test, y_pred, pos_label="Real"))
    return f1_score(y_test, y_pred, pos_label="Real")


def decisionModels(trainData, testData):
    # Separar label de la informacion y poner los datos en x_ y los labels en y_
    x_train, x_test, y_train, y_test = splitdataset(trainData, testData)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(x_train)
    x_train_imp = imp.transform(x_train)
    imp = imp.fit(x_test)
    x_test_imp = imp.transform(x_test)

    # Uso de multiples clasificadores para pruebas
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(criterion="gini", max_depth=5),
        DecisionTreeClassifier(criterion="entropy", max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree Gini", "Decision Tree Entropy", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    res = "Model Name --> Accuracy --> F1 Score\n"
    for i in range(len(classifiers)):
        clf = classifiers[i]
        name = names[i]
        clf.fit(x_train_imp, y_train)
        accur = clf.score(x_test_imp, y_test)
        y_pred = clf.predict(x_test_imp)
        res += name + " --> \t\t" + str(accur) + " ------> \t" + str(cal_accuracy(y_test, y_pred)) + "\n"
    return res


def createProfile(data, label, length, user):
    # Perfil basado en distancias 3D entre los puntos

    distances = []
    for i in range(length - 1):
        x_ = data[i].get('X')
        y_ = data[i].get('Y')
        z_ = data[i].get('Z')

        x_1 = data[i + 1].get('X')
        y_1 = data[i + 1].get('Y')
        z_1 = data[i + 1].get('Z')

        xS = (x_1 - x_) ** 2
        yS = (y_1 - y_) ** 2
        zS = (z_1 - z_) ** 2
        dist = (xS + yS + zS) ** 1 / 2
        distances.append(dist)

    return distances


@app.route('/Train', methods=['POST'])
def postAccData():
    # Recibir los valores del json
    rJson = request.get_json()
    accelData = rJson.get('Accel')
    user = rJson.get('User')
    dataLength = rJson.get('Length')
    dataLabel = rJson.get('Label')

    distanceData = createProfile(accelData, dataLabel, dataLength, user)

    dbMongoP = mongo.db.AccelData.Profile
    dbMongo = mongo.db.AccelData.Data

    dbMongoP1 = mongo.db.AccelData.ProfilePrueba
    dbMongo1 = mongo.db.AccelData.DataPrueba
    if dataLabel == "?":
        # crear el elemento e ingresarlo a la base de datos
        newEl = {'User': user, 'Length': dataLength, 'Profile': distanceData, 'Label': dataLabel}
        dbMongoP1.insert_one(newEl)
        # Tambien se guarda la informacion directa del acelerometro para pruebas
        newEl = {'User': user, 'Length': dataLength, 'AccelData': accelData, 'Label': dataLabel}
        dbMongo1.insert_one(newEl)

        testDataframe = pd.DataFrame(accelData)
        query = {'User': user}
        queryRes = dbMongoP.find_one(query, {'_id': 0})
        if queryRes is None:
            return "Usuario no existente"
        queryData = queryRes.get('Profile')
        trainDataframe = pd.DataFrame(queryData)

        #decisionModels(trainDataframe, testDataframe)
        return 'Modelo generado correctamente'
    else:
        # crear el elemento e ingresarlo a la base de datos
        newEl = {'User': user, 'Length': dataLength, 'Profile': distanceData, 'Label': dataLabel}
        dbMongoP.insert_one(newEl)
        # Tambien se guarda la informacion directa del acelerometro para pruebas
        newEl = {'User': user, 'Length': dataLength, 'AccelData': accelData, 'Label': dataLabel}
        dbMongo.insert_one(newEl)
        return 'Creado Exitosamente'


@app.route('/Train', methods=['GET'])
def getAccData():
    dbMongo = mongo.db.AccelData.Profile
    accelData = dbMongo.find()
    res = [{'User': acc['User'], 'Profile': acc['Profile'], 'Label': acc['Label']} for acc in accelData]

    return Response(json.dumps(res), mimetype='application/json')


@app.route('/Exists', methods=['POST', 'GET'])
def checkUserExistence():
    rJson = request.get_json()
    user = rJson.get('User')
    dbMongo = mongo.db.AccelData.Data
    query = {'User': user}
    queryRes = dbMongo.find_one(query, {'_id': 0})
    if queryRes is None:
        return 'None'
    return 'Exists'


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


@app.route('/Test', methods=['GET', 'POST'])
def testModels():
    rJson = request.get_json()
    user = rJson.get('User')
    number = rJson.get('Number')
    query = {'User': user}
    dbMongo = mongo.db.AccelData.Profile
    queryRes = dbMongo.find_one(query, {'_id': 0})

    queryData = queryRes.get('Profile')[:54]
    queryLabel = queryRes.get('Label')
    dataList = chunkIt(queryData, int(number))

    dataList_Label = []
    for i in range(len(dataList)):
        dataList_Label.append([queryLabel])
        for j in dataList[i]:
            dataList_Label[i].append(j)

    trainDataframe = pd.DataFrame(dataList_Label)

    dbMongo = mongo.db.AccelData.ProfilePrueba
    queryRes = dbMongo.find()
    dataList = []
    dataList_Label = []
    for i in queryRes:
        tempList = chunkIt(i['Profile'][:54], number)
        for x in tempList:
            dataList.append([i['User'], x, i['Label']])

    for i in range(len(dataList)):
        if dataList[i][0] == user:
            dataList_Label.append([queryLabel])
        else:
            dataList_Label.append([dataList[i][2]])

        for j in dataList[i][1]:
            dataList_Label[i].append(j)
    testDataframe = pd.DataFrame(dataList_Label)

    return "Number of partitions =" + str(number)+"\n"+decisionModels(testDataframe, trainDataframe)


@app.route('/', methods=['GET'])
def mainPage():
    return "This page is to see if the app is running correctly"


if __name__ == '__main__':
    app.run()
