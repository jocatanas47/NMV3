# klase imaju raylicit broj podataka - nebalansirane klase
# undersampling/oversampling ili dodeljivanje tezine klasama
# metrika - proveravamo performanse:
# - tacnost - koliko je odbiraka klasifikovano u odnosu na dataset #tacnih/#ukupno - losa kod nebalansiranih klasa
# -- konfuziona matrica
# - preciznost P = #TruePositive/(#TruePositive + #FalseNegative)
# - osetljivost R = #TruePositive/(#TruePositive + #FalsePositive)
# - F1 score = 2*P*R/(P + R)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import compute_class_weight

from keras import Sequential
from keras.layers import Dense

pod = pd.read_csv('podaciCas03.csv')
print(pod)

pod_trening, pod_test = train_test_split(pod, test_size=0.2, random_state=20)

ulaz_trening = pod_trening.drop(columns='d')
izlaz_trening = pod_trening.d
ulaz_test = pod_test.drop(columns='d')
izlaz_test = pod_test.d

# dataframe ne moze da se indeksira direktno uglastim yagradama
# koristimo loc - indeksiranje prema logickim vrednostima
K0 = ulaz_trening.loc[izlaz_trening==0, :]
K1 = ulaz_trening.loc[izlaz_trening==1, :]

plt.figure()
plt.plot(K0.iloc[:, 0], K0.iloc[:, 1], 'o')
plt.plot(K1.iloc[:, 0], K1.iloc[:, 1], 'x')
plt.show()

print(K0.shape)
print(K1.shape)

plt.figure()
izlaz_trening.hist()
plt.show()

def make_model(x):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=x.shape[1]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss='binary_crossentropy')

    return model

def calculate_metrics(ulaz_test, izlaz_test, model):
    # zaokruzijemo sigmoid
    pred = np.round(model.predict(ulaz_test))
    cm = confusion_matrix(izlaz_test, pred)
    TN = cm[0, 0]
    TP = cm[1, 1]
    FN = cm[0, 1]
    FP = cm[1, 0]
    acc = (TP + TN)/(TP + FP + FN + TN)
    P = TP/(FP + TP)
    R = TP/(FN + TP)
    F1 = 2*P*R/(P + R)
    print(acc)
    print(P)
    print(R)
    print(F1)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['K0', 'K1'])
    cm_display.plot()
    plt.show()

# Prvi nacin treeniranja - nema podesavanja sto se tice klasa
"""
nm = make_model(ulaz_trening)
nm.fit(ulaz_trening, izlaz_trening, epochs=1000, batch_size=ulaz_trening.shape[0], verbose=0)
calculate_metrics(ulaz_test, izlaz_test, nm)
"""

# Undersampling
K0us = K0.sample(K1.shape[0])
print(K0us.shape)

plt.figure()
plt.plot(K0us.iloc[:, 0], K0us.iloc[:, 1], 'o')
plt.plot(K1.iloc[:, 0], K1.iloc[:, 1], 'x')
plt.show()

# ignore_index - resetuje nesto
ulaz_trening_us = pd.concat((K0us, K1), axis=0, ignore_index=True)
izlaz_trening_us = pd.Series((np.append(np.zeros((K0us.shape[0], 1)),
                                        np.ones((K1.shape[0], 1)))))

"""
nm = make_model(ulaz_trening_us)
nm.fit(ulaz_trening_us, izlaz_trening_us, epochs=1000, batch_size=ulaz_trening.shape[0], verbose=0)
calculate_metrics(ulaz_test, izlaz_test, nm)
"""

# Oversampling - uvecavamo klasu 1
K1os = K1.sample(K0.shape[0], replace=True)

plt.figure()
plt.plot(K0.iloc[:, 0], K0.iloc[:, 1], 'o')
plt.plot(K1os.iloc[:, 0], K1os.iloc[:, 1], 'x')
plt.show()

ulaz_trening_os = pd.concat((K0, K1os), axis=0, ignore_index=True)
izlaz_trening_os = pd.Series((np.append(np.zeros((K0.shape[0], 1)),
                                        np.ones((K1os.shape[0], 1)))))

"""
nm = make_model(ulaz_trening_us)
nm.fit(ulaz_trening_os, izlaz_trening_os, epochs=1000, batch_size=ulaz_trening.shape[0], verbose=0)
calculate_metrics(ulaz_test, izlaz_test, nm)
"""

# dodeljivanje tezina klasama
weights = compute_class_weight(class_weight='balanced', classes=np.unique(izlaz_trening), y=izlaz_trening)
print(weights)

nm = make_model(ulaz_trening_us)
nm.fit(ulaz_trening_os, izlaz_trening_os, epochs=1000, batch_size=ulaz_trening.shape[0],
       class_weight={0: weights[0], 1: weights[1]}, verbose=0)
calculate_metrics(ulaz_test, izlaz_test, nm)