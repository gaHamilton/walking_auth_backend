from io import BytesIO
from flask import Flask, request, Response, send_file
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
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')

app = Flask(__name__)
MONGO_URL = "mongodb+srv://gHamilton:***REMOVED***@cluster0.j7fth.mongodb.net/myFirstDatabase?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"
app.config["MONGO_URI"] = MONGO_URL
mongo = PyMongo(app)
trueLabel = "Real"
falseLabel = "?"
bestPrecission = 0
bestF1Score = 0
bestWindowLength = 0
bestWindowDisplacement = 0


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
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # print("Accuracy : \n", accuracy_score(y_test, y_pred) * 100)

    # print("Report : \n", classification_report(y_test, y_pred))

    # print("F1 Score: \n", f1_score(y_test, y_pred, pos_label="Real"))
    return f1_score(y_test, y_pred, pos_label=trueLabel)


def decisionModels(trainData, testData, currentSize, currentStep):
    global bestWindowDisplacement, bestWindowLength, bestF1Score, bestPrecission
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

    res = "Model Name --> Accuracy --> F1 Score\n -------------------------------\n"

    values = []
    for i in range(len(classifiers)):
        clf = classifiers[i]
        name = names[i]
        clf.fit(x_train_imp, y_train)
        accur = clf.score(x_test_imp, y_test)
        values.append(accur)
        y_pred = clf.predict(x_test_imp)
        f1 = cal_accuracy(y_test, y_pred)
        if accur > bestPrecission and f1 > bestF1Score:
            bestPrecission = accur
            bestF1Score = f1
            bestWindowDisplacement = currentStep
            bestWindowLength = currentSize

        res += name + " --> \t\t" + str(accur) + " ------> \t" + str(f1) + "\n"
    top3 = sorted(zip(values,names),reverse=True)[:3]
    for i in range(len(top3)):
        el = top3[i]
        modelsDict[el[1]] = modelsDict[el[1]] +1
        if i == 0:
            top1Ape[el[1]] = top1Ape[el[1]] +1
        elif i ==1 :
            top2Ape[el[1]] = top2Ape[el[1]] + 1
        elif i==2:
            top3Ape[el[1]] = top3Ape[el[1]] + 1
    # res += "Best value: "+str(best)
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


def checkExists(user):
    dbMongo = mongo.db.AccelData.Profile
    query = {'User': user}
    queryRes = dbMongo.find_one(query, {'_id': 0})
    if queryRes is None:
        return 'None'
    return queryRes


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
        ans = checkExists(user)
        if ans == 'None':
            return "Usuario no existente"
        queryData = ans.get('Profile')
        trainDataframe = pd.DataFrame(queryData)

        # TODO respuesta del modelo en la aplicacion?
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
    if checkExists(user) == 'None':
        return 'None'
    return 'Exists'


def windows(list, length, move):
    out = []
    for i in range(0, len(list) - (length - move), move):
        out.append(list[i:i + length])
    return out


def unifyFreq(dataList, minSize):
    #print("Unify")
    steps = int(len(dataList) / minSize)
    out = []
    for i in range(0, len(dataList), steps):
        if len(out) >= minSize:
            break
        out.append(dataList[i])
    return out


def cutFreq(dataList, newListSize):
    #print("Cut")
    return dataList[:newListSize]


def findSmallest(dataList):
    resList = []
    res = np.inf
    for i in dataList:
        resList.append(i)
        length = int(len(i['Profile']))
        if length < res:
            res = length
    return resList, res


@app.route('/Test', methods=['GET', 'POST'])
def testModels():
    global bestWindowDisplacement, bestWindowLength, bestF1Score, bestPrecission, modelsDict, top3Ape, top2Ape, top1Ape
    modelsDict = {"Nearest Neighbors": 0, "Linear SVM": 0, "RBF SVM": 0, "Gaussian Process": 0,
                  "Decision Tree Gini": 0, "Decision Tree Entropy": 0, "Random Forest": 0, "Neural Net": 0,
                  "AdaBoost": 0,
                  "Naive Bayes": 0, "QDA": 0}
    top1Ape = {"Nearest Neighbors": 0, "Linear SVM": 0, "RBF SVM": 0, "Gaussian Process": 0,
               "Decision Tree Gini": 0, "Decision Tree Entropy": 0, "Random Forest": 0, "Neural Net": 0, "AdaBoost": 0,
               "Naive Bayes": 0, "QDA": 0}
    top2Ape = {"Nearest Neighbors": 0, "Linear SVM": 0, "RBF SVM": 0, "Gaussian Process": 0,
               "Decision Tree Gini": 0, "Decision Tree Entropy": 0, "Random Forest": 0, "Neural Net": 0, "AdaBoost": 0,
               "Naive Bayes": 0, "QDA": 0}
    top3Ape = {"Nearest Neighbors": 0, "Linear SVM": 0, "RBF SVM": 0, "Gaussian Process": 0,
               "Decision Tree Gini": 0, "Decision Tree Entropy": 0, "Random Forest": 0, "Neural Net": 0, "AdaBoost": 0,
               "Naive Bayes": 0, "QDA": 0}

    bestPrecission = 0
    bestF1Score = 0
    bestWindowLength = 0
    bestWindowDisplacement = 0

    rJson = request.get_json()
    user = rJson.get('User')
    size = rJson.get('size')
    step = rJson.get('steps')
    freqHandle = rJson.get('UnifyFrequency')
    simplifiedVersion = rJson.get("Simplify")

    if checkExists(user) == 'None':
        return "Usuario no existente"

    # Adquisicion de los datos para entrenar de la base de datos
    dbMongo = mongo.db.AccelData.Profile
    queryRes = dbMongo.find()

    # Adquisicion de los datos a probar de la base de datos
    dbMongoPrueba = mongo.db.AccelData.ProfilePrueba
    query = {'User': user}
    queryResP = dbMongoPrueba.find_one(query, {'_id': 0})
    queryDataP = queryResP.get('Profile')

    queryNegRes = dbMongoPrueba.find_one({'User':"negTest"})

    queryList, minSize = findSmallest(queryRes)

    if freqHandle:
        queryNegList = unifyFreq(queryNegRes['Profile'], minSize)
        queryListP = unifyFreq(queryDataP, minSize)
    else:
        queryNegList = cutFreq(queryNegRes['Profile'], minSize)
        queryListP = cutFreq(queryDataP, minSize)
    finRes = ""

    for si in size:
        for ste in step:
            if si == 5 and ste >5:
                continue
            dataList = []
            dataList_Label = []
            for i in queryList:

                if i['User'] == "negTest":
                    continue

                # Debido a los diferentes tamanios de sets de datos, unificar o cortar al mas pequenio en los obtenidos
                # en la base de datos para que todos tengan el mismo tamanio para ser manajeados por los modelos
                if freqHandle:
                    unifiedListSize = unifyFreq(i['Profile'], minSize)
                else:
                    unifiedListSize = cutFreq(i['Profile'], minSize)

                # Una vez se tienen los datos del mismo tamanio, crear ventanas
                tempList = windows(unifiedListSize, si, ste)

                # se le aniade el usuario y label a los datos de la ventanas creando una nueva lista del estilo:
                # [[Usuario, ventana1, etiqueta],
                # [Usuario, ventana2, etiqueta], ...]
                for x in tempList:
                    dataList.append([i['User'], x, falseLabel])

            for i in range(len(dataList)):
                # Si el usuario es el que se va a entrenar a detectar, usa el label de 'Real', de lo contrario usan el label "?"
                if dataList[i][0] == user:
                    dataList_Label.append([trueLabel])
                else:
                    dataList_Label.append([dataList[i][2]])

                # Ahora que se tiene el label de los datos, se le aniaden los datos de la ventana, tiene que quedar en una
                # lista antes de usar pandas o va a quedar como una lista de datos distitna
                for j in dataList[i][1]:
                    dataList_Label[i].append(j)

            # Pandas para dejar los datos en el formato adecuado a usar por los modelos
            trainDataframe = pd.DataFrame(dataList_Label)
            # print(trainDataframe)

            # Se repite el proceso, solo que ahora se usa solo el perfil del usuario a detectar y un perfil que siempre deberia marcar como '?'

            dataList = windows(queryListP, si, ste)
            data2List = windows(queryNegList, si,ste)

            dataNegList = []
            dataList_Label = []
            for i in range(len(dataList)):
                dataList_Label.append([trueLabel])
                dataNegList.append([falseLabel])
                for j in dataList[i]:
                    dataList_Label[i].append(j)
                for j in data2List[i]:
                    dataNegList[i].append(j)
            testDataframe = pd.DataFrame(dataList_Label+dataNegList)
            #print(testDataframe)

            finRes = finRes + "User: " + user + "\nWindow Size =" + str(si) + "\n Window displacement= " + str(ste) \
                     + "\n" + decisionModels(trainDataframe, testDataframe, si, ste) + "\n\n\n"
    simply = "User: "+user+"\nBest Precision: "+str(bestPrecission)+"\nBest F1 Score: "+str(bestF1Score)+"\nWindow Lenght: "+str(bestWindowLength)+"\nWindow Displacement: "+str(bestWindowDisplacement)+"\n"
    if simplifiedVersion:
        #print(finRes)

        return str(modelsDict)+"\n#1: "+str(top1Ape)+"\n#2: "+str(top2Ape)+"\n#3: "+str(top3Ape)+"\n\n"+simply
    #print(modelsDict)
    #print("T1"+str(top1Ape))
    #print("T2"+str(top2Ape))
    #print("T3"+str(top3Ape))
    #print(simply)
    return finRes


@app.route('/Paint', methods=['GET'])
def paintLocal():
    # Recibir el nombre de usuario que se va a graficar
    rJson = request.get_json()
    user = rJson.get('User')
    dbMongo = mongo.db.AccelData.Profile
    query = {'User': user}
    queryRes = dbMongo.find_one(query, {'_id': 0})
    if queryRes is None:
        return 'None'

    fig = plt.figure()
    data = queryRes.get('Profile')
    x_L = [i for i in range(len(data))]
    plt.plot(x_L, data)
    plt.title("Perfil De Usuario")
    plt.ylabel("Distancias")
    plt.xlabel("# Dato")
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    return send_file(img, as_attachment=True, attachment_filename="Intersection.png")


@app.route('/', methods=['GET'])
def mainPage():
    return "This page is to see if the app is running correctly"


if __name__ == '__main__':
    app.run()
