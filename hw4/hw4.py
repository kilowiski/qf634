# Importing necessary Libraries
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import initializers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

keras.utils.set_random_seed(5)
tf.config.experimental.enable_op_determinism()

# Loading Dataset
data = pd.read_csv("Churn_Modelling.csv")

# Generating Dependent Variable Vectors
Y = data.iloc[:, -1].values
X = data.iloc[:, 3:13]
X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})

# Encoding Categorical variable Geography


ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Performing Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# NNmodel
NNmodel = tf.keras.models.Sequential()
NNmodel.add(tf.keras.layers.Dense(units=6, activation="sigmoid"))
NNmodel.add(tf.keras.layers.Dense(units=6, activation="sigmoid"))
NNmodel.add(tf.keras.layers.Dense(units=6, activation="sigmoid"))
NNmodel.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
NNmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = NNmodel.fit(X_train, Y_train, batch_size=100, epochs=500)
### Now we use the trained NNmodel to predict output in X_train sample
eval_train = NNmodel.evaluate(
    X_train, Y_train, return_dict=True
)  ### evaluates the loss and accuracy as specified in the Compiler
print(eval_train)

import matplotlib.pyplot as plt

print(history.history["accuracy"])
print(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("model accuracy")
plt.ylabel("accuracy and loss")
plt.xlabel("epoch")
plt.legend(["accuracy", "loss"], loc="upper left")
plt.show()

newdata = sc.transform([[0, 0, 1, 650, 1, 60, 2, 300000, 2, 1, 0, 80000]])
print(NNmodel.predict(newdata))
