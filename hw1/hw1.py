import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score


# ok lah the prof made some functions that we can use
def retreive_train_results(X_train, y_train, reg_obj):
    y_pred_train = reg_obj.predict(X_train)
    r2_score_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    return {"r2_score_train": r2_score_train, "rmse_train": rmse_train}


def retreive_test_results(X_test, y_test, reg_obj):
    def tss(y_test):
        return ((y_test - np.mean(y_test)) ** 2).sum()

    y_pred = reg_obj.intercept_ + np.dot(X_test, reg_obj.coef_.T)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    # can just use r2_score_Linreg_test = r2_score(y_test, y_pred) actually
    r2_score_test = 1 - (len(y_pred) * (rmse_test) ** 2) / tss(y_test)
    return {"r2_score_test": r2_score_test, "rmse_test": rmse_test}


df = pd.read_csv("/home/kilo/Desktop/qf634/qf634file/cruise_ship_info.csv")
df.hist(bins=50, figsize=(20, 15))
# visually a candidate for standard scaling
plt.show()

# scale using standard scaler !
standard_scaler = preprocessing.StandardScaler()
remove = ["Ship_name", "Cruise_line", "crewx100"]
feature_cols = [x for x in df.columns if x not in remove]

# feature and target
X = df.loc[:, feature_cols]
y = df["crewx100"]
# actually use the scaler here
X = pd.DataFrame(data=standard_scaler.fit_transform(X), columns=feature_cols)

# plot the scaled and reg
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
axs = axs.flatten()  ## returns flatted version of array
for i, k in enumerate(feature_cols):
    # Plot data and a linear regression model fit.
    sns.regplot(y=y, x=X[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

# actual start of the reg
# Split the 158 rows of sample into 70% training and 30% test. In train_test_split, use random_state = 0
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=0
)


# do linreg here
Linreg = LinearRegression()
Linreg.fit(X_train, y_train)
print("Linreg results")
print(retreive_train_results(X_train, y_train, Linreg))
print(retreive_test_results(X_test, y_test, Linreg))

# do ridge here
Ridge = Ridge(alpha=0.05)
Ridge.fit(X_train, y_train)
print("Ridge Results")
print(retreive_train_results(X_train, y_train, Ridge))
print(retreive_test_results(X_test, y_test, Ridge))

# do lasso here
Lasso = Lasso(alpha=0.01)
Lasso.fit(X_train, y_train)
print("Lasso Results")
print(retreive_train_results(X_train, y_train, Lasso))
print(retreive_test_results(X_test, y_test, Lasso))

# shuffling
X_shuffle, y_shuffle = shuffle(X_train, y_train, random_state=0)
### Rename X_Shuffle y_shuffle
X_train = X_shuffle
y_train = y_shuffle

# cross validation with LINEAR REG
### This combined training set X_train, y_train is split into k=4 (cv=4) folds for each of k=1,2,...,4 repetitions
### Score is R2 measure, there are 4 scores since k=cv=4, one for each repetition
scoresLinreg = cross_val_score(estimator=Linreg, X=X_train, y=y_train, cv=4)
print("---------------------------------------------")
print('linreg scores, CV 4')
print(scoresLinreg)
print(
    "\Linreg 4-fold cross validation: %0.4f mean R2 with a standard deviation of %0.4f"
    % (scoresLinreg.mean(), scoresLinreg.std())
)

# cross validation with RIDGE REG
scoresRidge = cross_val_score(estimator=Ridge, X=X_train, y=y_train, cv=4)
print("---------------------------------------------")
print('ridge scores, CV 4')
print(scoresRidge)
print(
    "Ridge 4-fold cross validation:%0.4f mean R2 with a standard deviation of %0.4f"
    % (scoresRidge.mean(), scoresRidge.std())
)
