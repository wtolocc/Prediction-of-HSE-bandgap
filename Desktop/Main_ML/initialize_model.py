#2021.06.02
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split
from load_store_dataframe import load_dataframe_from_json
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def model1_traning_classifier_3d():
    model = RandomForestClassifier()

    data = load_dataframe_from_json('model/data_feature_for_classify.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_class_metalORnometal_e.txt', 'rb'))
    features = features_rank[:21]

    X = data[features]
    y = data['is semi'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model, 'model/model_classify_metalORnometal.pkl')

def model1_traning_3d():

    model = RandomForestRegressor(
                                       max_depth=29,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       n_estimators=240)

    data=load_dataframe_from_json('model/data_feature_e.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_e.txt', 'rb'))
    features = features_rank[:30]

    X = data[features]
    y = data['Band_gap_HSE'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model,'model/model_e.pkl')

def model2_traning_3d():

    data=load_dataframe_from_json('model/data_feature_e.json')
    data=data[data['Band_gap_GGA']>0.7]

    X = data['Band_gap_GGA'].values.reshape(-1,1)
    y = data['Band_gap_HSE'].values

    model = LinearRegression()
    model.fit(X,y)
    #print("linear expression: {}·X+{}".format(model.coef_[0],model.intercept_))
    joblib.dump(model,'model/model_linear.pkl')

def model3_traning_3d():

    model = RandomForestRegressor(
                                       max_depth=29,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       n_estimators=360)

    data=load_dataframe_from_json('model/data_feature_e.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_eANDgap.txt', 'rb'))
    features = features_rank[:22]

    X = data[features]
    y = data['Band_gap_HSE'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model,'model/model_eANDgap.pkl')

def model1_traning_classifier_2d():

    model = GradientBoostingClassifier(learning_rate=0.2,max_depth=4,min_samples_leaf=2,min_samples_split=2,
                                       n_estimators=70)

    data = load_dataframe_from_json('model/data_feature_c2db.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_class_metalORnometal_c2db_e.txt', 'rb'))
    features = features_rank[:14]

    X = data[features]
    y = data['is semi'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model, 'model/model_classify_metalORnometal_c2db.pkl')

def model1_traning_2d():

    model = RandomForestRegressor(
                                       max_depth=29,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       n_estimators=240)

    data=load_dataframe_from_json('model/data_feature_c2db.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_c2db_e.txt', 'rb'))
    features = features_rank[:16]

    X = data[features]
    y = data['gap_hse'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model,'model/model_c2db_e.pkl')

def model2_traning_2d():

    data=load_dataframe_from_json('model/data_feature_c2db.json')


    X = data['Band_gap_GGA'].values.reshape(-1,1)
    y = data['gap_hse'].values

    model = LinearRegression()
    model.fit(X,y)
    #print("linear expression: {}·X+{}".format(model.coef_[0],model.intercept_))
    joblib.dump(model,'model/model_c2db_linear.pkl')

def model3_traning_2d():

    model = RandomForestRegressor(
                                       max_depth=29,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       n_estimators=360)

    data=load_dataframe_from_json('model/data_feature_c2db.json')
    data = data.dropna()

    features_rank = pickle.load(open('model/features_rank_c2db_eANDgap.txt', 'rb'))
    features = features_rank[:21]

    X = data[features]
    y = data['gap_hse'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    joblib.dump(model,'model/model_c2db_eANDgap.pkl')

if __name__ == '__main__':
    # model1_traning_classifier_3d()
    # model1_traning_3d()
    # model2_traning_3d()
    # model3_traning_3d()
    model1_traning_classifier_2d()
    # model1_traning_2d()
    # model2_traning_2d()
    # model3_traning_2d()