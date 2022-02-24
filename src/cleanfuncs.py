import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor as KNNR


#####################################################
def roomtype():
    '''
    Returns the original column 'room_type' from the fullset as imported
    (This function only works for 'train.csv' in the same directory as the .ipynb)
    file
    '''
    fullset = pd.read_csv('train.csv')
    return fullset.room_type

#################################################
def clean10_mercat(x):
    numset = x._get_numeric_data()
    numset = numset.drop(['calendar_updated','bathrooms','neighbourhood_group_cleansed',
                          'scrape_id','host_id','availability_30','availability_60',
                        'availability_90','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d',
                        'calculated_host_listings_count','calculated_host_listings_count_entire_homes',        
                        'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms', 
                        'host_total_listings_count','minimum_nights','minimum_nights_avg_ntm',
                        'maximum_minimum_nights','maximum_nights_avg_ntm',
                        'reviews_per_month', 'id','review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness',                    
                        'review_scores_checkin','review_scores_communication','review_scores_location',                          
                        'review_scores_value' ], axis=1)
    return numset
#################################################
def clean10(x):
    numset = x._get_numeric_data()
    numset = numset.drop(['calendar_updated','bathrooms','neighbourhood_group_cleansed',
                          'scrape_id','host_id','latitude','longitude','availability_30','availability_60',
                        'availability_90','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d',
                        'calculated_host_listings_count','calculated_host_listings_count_entire_homes',        
                        'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms', 
                        'host_total_listings_count','minimum_nights','minimum_nights_avg_ntm',
                        'maximum_minimum_nights','maximum_nights_avg_ntm',
                        'reviews_per_month', 'id','review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness',                    
                        'review_scores_checkin','review_scores_communication','review_scores_location',                          
                        'review_scores_value'], axis=1)
    return numset
###################################################
def prueba_modelo(modelo):

    modelo.fit(X_train, y_train)

    y_pred=modelo.predict(X_test)

    train_score=modelo.score(X_train, y_train)  
    test_score=modelo.score(X_test, y_test)
    
    print(modelo)
    print('Train:', train_score)
    print('Test:', test_score) 
    print('\n')
####################################################    
def bosque(n):
    rfr=RFR(n_estimators=n)
    rfr.fit(X_train, y_train)

    y_pred=rfr.predict(X_test)

    train_score=rfr.score(X_train, y_train)  
    test_score=rfr.score(X_test, y_test)


    print('Train:', train_score)
    print('Test:', test_score) 
#####################################################
def distance(geoset):
    '''
    Accepts a single dataframe as argument.
    This function takes a pair of coordinates from a dataframe and measures its distance
    from a predefined set of coordinates known as 'benchmark'. Then returns a list containing
    all such distances ready to become a new df column
    '''
    from geopy.distance import geodesic
    import math
    distance = []
    benchmark = (52.360486, 4.885896) #Rijksmuseum
    for i in range(0,len(geoset)):
        coordinates_i = (geoset.iloc[i]['latitude'],geoset.iloc[i]['longitude'])

        if (geodesic(coordinates_i, benchmark).kilometers) < 100:
            distance.append(math.exp(geodesic(coordinates_i, benchmark).kilometers))
        else:
            distance.append(100)

        
        
    return distance
#####################################################
def bathroom_cleanse(x):
    ''' 
    Función ad-hoc para limpiar la columna de los baños
    '''
    import re
    if x == np.NaN:
        return 0
    elif '0' in x:
        return 0
    elif (re.findall('(\d\.\d)', x)):
        if 'shared' in x:
            #aquí dividiste entre dos pero bien podrías tenerlo a secas o hacer otro invento con el shared
            return float(re.findall('(\d\.\d)', x)[0])/2
        elif 'Half-bath' in x:
            return 0.5
        else:
            return float(re.findall('(\d\.\d)', x)[0])
    else:
        if 'shared' in x:
            #aquí dividiste entre dos pero bien podrías tenerlo a secas o hacer otro invento con el shared
            return float(re.findall('(\d)', x)[0])/2
        elif 'Half-bath' in x:
            return 0.5
        elif 'Shared half-bath' in x:
            return 0.25
        elif 'Private half-bath' in x:
            return 0.5
        else:
            return float(re.findall('(\d)', x)[0])
#####################################################
def tf_01(x):
    '''
    Si recibe 'f' retorna 0, y si recibe 't' retorna 1
    '''
    if x == 'f':
        return 0
    if x == 't':
        return 1
#####################################################
def geobinary(x):
    if x > 4: #subido de 3 a 5
        return 1
    else:
        return 0