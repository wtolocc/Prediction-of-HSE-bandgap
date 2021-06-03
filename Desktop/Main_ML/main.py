import pickle
import joblib
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
import pandas as pd
from load_store_dataframe import store_dataframe_as_json, load_dataframe_from_json

def model_classifier_3d(formula):
    data = pd.DataFrame({'formula': [formula]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_class_metalORnometal_e.txt', 'rb'))
    features = features_rank[:21]
    # 读取模型
    model = joblib.load('model/model_classify_metalORnometal.pkl')
    X = data[features]
    data['is semi'] = model.predict(X)
    return data['is semi'].values[0]

def model_1_3d(formula):
    data = pd.DataFrame({'formula': [formula]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_e.txt', 'rb'))
    features = features_rank[:30]
    # 读取模型
    model = joblib.load('model/model_e.pkl')
    X = data[features]
    data['predict_Band_gap_HSE'] = model.predict(X)
    print('formula : {}\nHSE Bandgap: {}'
          .format(data['formula'].values[0],data['predict_Band_gap_HSE'].values[0])
        )

def model_2_3d(bandgap_pbe):
    predicted_Band_gap_HSE=1.1548885344578868*float(bandgap_pbe)+0.8421296188748237
    print('formula : None\nHSE Bandgap: {}'.format(predicted_Band_gap_HSE))

def model_3_3d(formula,bandgap_pbe):
    data = pd.DataFrame({'formula': [formula],
                         'Band_gap_GGA': [float(bandgap_pbe)]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_eANDgap.txt', 'rb'))
    features = features_rank[:22]
    # 读取模型
    model = joblib.load('model/model_eANDgap.pkl')
    X = data[features]
    data['predict_Band_gap_HSE'] = model.predict(X)
    print('formula : {}\nHSE Bandgap: {}'
          .format(data['formula'].values[0], data['predict_Band_gap_HSE'].values[0])
          )

def main_ML_3d(formula,bandgap_pbe):
    if  formula:
        if bandgap_pbe:
            model_3_3d(formula,bandgap_pbe)
            print('The model has score=0.96,RMSE=0.28')
        else:
            res=model_classifier_3d(formula)
            if res>0:
                model_1_3d(formula)
                print('The model has score=0.76,RMSE=0.75')
            else:
                print('formula : {}\n Bandgap: 0'.format(formula))
    else:
        model_2_3d(bandgap_pbe)
        print('The model has score=0.92,RMSE=0.43')

def model_classifier_2d(formula):
    data = pd.DataFrame({'formula': [formula]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_class_metalORnometal_c2db_e.txt', 'rb'))
    features = features_rank[:14]
    # 读取模型
    model = joblib.load('model/model_classify_metalORnometal_c2db.pkl')
    X = data[features]
    data['is semi'] = model.predict(X)
    return data['is semi'].values[0]

def model_1_2d(formula):
    data = pd.DataFrame({'formula': [formula]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_c2db_e.txt', 'rb'))
    features = features_rank[:16]
    # 读取模型
    model = joblib.load('model/model_c2db_e.pkl')
    X = data[features]
    data['predict_Band_gap_HSE'] = model.predict(X)
    print('formula : {}\nHSE Bandgap: {}'
          .format(data['formula'].values[0],data['predict_Band_gap_HSE'].values[0])
        )

def model_2_2d(bandgap_pbe):
    predicted_Band_gap_HSE=1.2135*float(bandgap_pbe)+0.6185
    print('formula : None\nHSE Bandgap: {}'.format(predicted_Band_gap_HSE))

def model_3_2d(formula,bandgap_pbe):
    data = pd.DataFrame({'formula': [formula],
                         'Band_gap_GGA': [float(bandgap_pbe)]})

    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj', ignore_errors=True)

    ##
    features_rank = pickle.load(open('model/features_rank_c2db_eANDgap.txt', 'rb'))
    features = features_rank[:21]
    # 读取模型
    model = joblib.load('model/model_c2db_eANDgap.pkl')
    X = data[features]
    data['predict_Band_gap_HSE'] = model.predict(X)
    print('formula : {}\nHSE Bandgap: {}'
          .format(data['formula'].values[0], data['predict_Band_gap_HSE'].values[0])
          )

def main_ML_2d(formula,bandgap_pbe):
    if  formula:
        if bandgap_pbe:
            model_3_2d(formula,bandgap_pbe)
            print('The model has score=0.94,RMSE=0.36')
        else:
            res=model_classifier_2d(formula)
            if res>0:
                model_1_2d(formula)
                print('The model has score=0.73,RMSE=0.80')
            else:
                print('formula : {}\n Bandgap: 0'.format(formula))
    else:
        model_2_2d(bandgap_pbe)
        print('The model has score=0.87,RMSE=0.54')

def main_ML(dimension,formula,bandgap_pbe):
    if dimension=='2d':
        main_ML_2d(formula,bandgap_pbe)
    elif dimension=='3d':
        main_ML_3d(formula, bandgap_pbe)

if __name__ == '__main__':
    dimension=input('dimension (2d or 3d) : ')
    formula=input('formula : ')
    bandgap_pbe=input('bandgap_pbe : ')
    main_ML(dimension,formula,bandgap_pbe)






