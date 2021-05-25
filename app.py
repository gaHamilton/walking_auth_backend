from flask import Flask, request, Response
from flask_pymongo import PyMongo
import json
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

app = Flask(__name__)
MONGO_URL = "mongodb+srv://gHamilton:***REMOVED***@cluster0.j7fth.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
app.config["MONGO_URI"] = MONGO_URL
mongo = PyMongo(app)


# Function to split the dataset
def splitdataset(trainData, testData):
    # Separating the target variable
    x_train = trainData.values[:, 1:5]
    x_test = testData.values[:, 1:5]
    y_train = trainData.values[:, 0]
    y_test = testData.values[:, 0]

    return x_train, x_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(x_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(x_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def train_using_entropy(x_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(x_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)

    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print("Report : ", classification_report(y_test, y_pred))


def decisionTree(trainData, testData):
    # Separar label de la informacion y poner los datos en x_ y los labels en y_
    x_train, x_test, y_train, y_test = splitdataset(trainData, testData)
    clf_gini = train_using_gini(x_train, y_train)
    clf_entropy = train_using_entropy(x_train, y_train)

    # Operational Phase

    # Prediction using gini
    y_pred_gini = prediction(x_test, clf_gini)
    #cal_accuracy(y_test, y_pred_gini)

    dbMongo = mongo.db.AccelData.ModelResponse
    newEl = {'PredictedValues': y_pred_gini, 'Method': 'Gini'}
    dbMongo.insert_one(newEl)

    # print("Results Using Entropy:")
    # Prediction using entropy
    # y_pred_entropy = prediction(x_test, clf_entropy)
    # cal_accuracy(y_test, y_pred_entropy)


@app.route('/Train', methods=['POST'])
def postAccData():
    # Recibir los valores del json
    rJson = request.get_json()
    accelData = rJson.get('Accel')
    user = rJson.get('User')
    dataLength = rJson.get('Length')
    dataLabel = rJson.get('Label')

    if dataLabel == "?":
        testDataframe = pd.DataFrame(accelData, columns=['Label','X', 'Y', 'Z'])
        query = {'User': user}
        dbMongo = mongo.db.AccelData.Data
        queryRes = dbMongo.find_one(query,{'_id':0})
        if queryRes is None:
            return "Usuario no existente"
        queryData = queryRes.get('AccelData')
        trainDataframe = pd.DataFrame(queryData, columns=['Label','X', 'Y', 'Z'])

        decisionTree(trainDataframe, testDataframe)
        return 'Modelo generado correctamente'
    else:
        # crear el elemento e ingresarlo a la base de datos
        dbMongo = mongo.db.AccelData.Data
        newEl = {'User': user, 'AccelData': accelData, 'Length': dataLength, 'Label': dataLabel}
        dbMongo.insert_one(newEl)
        return 'Creado Exitosamente'


@app.route('/Train', methods=['GET'])
def getAccData():
    dbMongo = mongo.db.AccelData.Data
    accelData = dbMongo.find()
    res = [{'user_id': acc['user_id'], 'AccelData': acc['AccelData'], 'Label': acc['Label']} for acc in accelData]
    return Response(json.dumps(res), mimetype='application/json')


if __name__ == '__main__':
    app.run()
