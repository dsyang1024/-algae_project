# import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def data_receive(filename):
    global df, XvarNames
    # Read in data
    df = pd.read_csv(filename+".csv")

    # Check columns
    print(df.columns)
    XvarNames = np.array(df.columns[1:])

    # Read info in case of NULL data
    # non-null means : no null data in the column
    print(df.info())
    # Print sum of the null data in the dataframe
    print(df.isnull().sum())

    # Visualize correlation
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()
    print("Check input data and correlation")

def normalize():
    global Y, X, X_train, X_val, Y_train, Y_val
    # Target vector
    target = df.columns[0]
    print(target)
    Y = df[target]

    # Feature matrix
    features = df.columns[1:]
    print(features)
    X = df[features]

    # Split data between train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=777)
    # Normalization
    from sklearn.preprocessing import StandardScaler
    sc= StandardScaler()

    # X_train = sc.fit_transform(X_train)
    # X_val = sc.fit_transform(X_val)
    # (Korean) 주성분 분석(PCA)를 통해 차원축소, 낮은 차원의 데이터를 찾아낸다
    from sklearn.decomposition import PCA
    pca = PCA(n_components=len(features))

    X_train = pca.fit_transform(X_train)
    X_val = pca.fit_transform(X_val)
    print("Data assignment done")

def train():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error


    nTree = range(50, 1000, 50)
    mselist = []

    for iTree in nTree:
        depth = None
        regressor = RandomForestRegressor(n_estimators=iTree, random_state=777,
                                          max_depth=depth)
        regressor.fit(X_train, Y_train)
        melb_preds = regressor.predict(X_val)
        print(iTree, "::", mean_absolute_error(Y_val, melb_preds))
        mselist.append(mean_absolute_error(Y_val, melb_preds))

    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    plt.plot(nTree, mselist)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Mean Squared Error')
    # plot.ylim([0.0, 1.1*max(mseOob)])
    plt.show()

    # 피처 중요도 도표 그리기
    featureImportance = regressor.feature_importances_

    # 가장 높은 중요도 기준으로 스케일링
    featureImportance = featureImportance / featureImportance.max()
    sorted_idx = np.argsort(featureImportance)
    barPos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(barPos, featureImportance[sorted_idx], align='center')
    plt.yticks(barPos, XvarNames[sorted_idx])
    plt.xlabel('Variable Importance')
    plt.show()