# -*- coding: utf-8 -*-
"""

### Part 2: Recommender system and predictions


*Author:* Jaime Fanjul.

In this section we curate, reformat, clean and impute the data in order to be used in the creation of the recommender system models.<br>

## Index:

1. [Load preprocessed data](#1)<br>
2. [User based model](#2)
3. [Product based model](#3)
4. [Most popular product model](#4)
5. [Workflow](#5)
6. [Evaluate models](#6)
7. [Save predictions](#7)

##### Imports needed
"""

#Manejo de data frames
import pandas as pd
# Procesamiento matemático
import numpy as np
# Gráficas
import matplotlib.pyplot as plt
# Métodos de imputacion
import sklearn.impute as si
# Visualización de datos faltantes
import missingno as msno
# Control de datos faltantes categoricos
import sklearn.preprocessing as sp
# Gráfico de cajas y bigotes
import seaborn as sns
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from fancyimpute import KNN
import scipy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

"""#### Functions needed"""

def get_cosine_similarity_matrix(df):
    """
    This function calculates and returns a cosine similarity matrix between the entries of the index of the df provided    
    """
    sparce_matrix = scipy.sparse.csr_matrix(df.values)
    similarities = cosine_similarity(sparce_matrix)
    cosine_sim_matrix = pd.DataFrame(similarities, index= df.index, columns = df.index)
    return cosine_sim_matrix


# USER BASED MODEL FUNCTIONS

def make_dummies(df, threshold = 0.01):
    cat_vars = df.select_dtypes(include=['object']).columns
    size_df = len(df)
    count = 0

    for col_name in cat_vars:
        dummed_features = []
        #print('The column smart encoded is: ',col_name)
        series_col = df[col_name].value_counts()
        indexes = series_col.index
        for index in indexes:
            ratio = series_col.get(index)/size_df
          #print('el index analizado es: '+ str(index)+ ' con un ratio de '+ str(ratio))
          
            if ratio < threshold:
                dummed_features.append(index)
                count = count + 1
                #print('sustituir num: ' + str(count))  
          
        print('the dummed features:')
        print(dummed_features)
        df.loc[df[col_name].isin(dummed_features), col_name] = col_name + '_others'
        df_intelligent_dummies = pd.get_dummies(df)


    return df_intelligent_dummies

# PRODUCT BASED MODEL FUNCTIONS




# MOST POPULAR BASED FUNCTIONS

def recommendations_by_most_popular(users_2_recommend, most_popular_prods, 
                                    df_mask_products,num_recommend = 1):
    
    d = ['ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']
    
    user_2_mask = df_mask_products.index.to_list() #This are the ones that we know they have bought something
                                                        #last month
    
    rec_most_pop = {}
    for user in users_2_recommend:
        N_indexes = np.zeros(25, dtype=int)
        
        if user in user_2_mask:
            most_common_recommendation = raw_products_2_recommend(most_popular_prods, 
                                                                  np.array(df_mask_products.loc[user]))
        else: 
            most_common_recommendation = raw_products_2_recommend(most_popular_prods, 
                                                                  np.ones(25, dtype=int))
        #print('most common masked:')
        #print(most_common_recommendation)
        for i in range(num_recommend):
            index_min = masked_argmin(most_common_recommendation, limit = 0)
            #index_min = np.argmin(most_common_recommendation * [most_common_recommendation!=0])
            #print('Recomment prod_' + str(index_min+1) +' ranking' +str(most_common_recommendation[index_min]))
            most_common_recommendation[index_min] = 0
            N_indexes[index_min] = 1
            
        #print('Recommendations for user:' +str(user))
        #print(N_indexes)
        rec_most_pop[user] = N_indexes 
    
    df_recommendation_by_most_popular = pd.DataFrame.from_dict(rec_most_pop, orient='index', columns=d)
    
    return df_recommendation_by_most_popular.rename_axis("cod_persona")


def mask_recommendations(users_2_recommend, raw_recommendations, 
                                    df_mask_products):
    
    d = ['ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']
    
    user_2_mask = df_mask_products.index.to_list() #This are the ones that we know they have bought something
                                                        #last month
    rec = {}
    
    for user in users_2_recommend:
        
        if user in user_2_mask:
            recommendation = raw_products_2_recommend(raw_recommendations.loc[user], 
                                                                  np.array(df_mask_products.loc[user]))
        else: 
            recommendation = raw_products_2_recommend(raw_recommendations.loc[user], 
                                                                  np.ones(25, dtype=int))
        
        rec[user] = recommendation 
    
    df_recommendation = pd.DataFrame.from_dict(rec, orient='index', columns=d)
    
    return df_recommendation.rename_axis("cod_persona")




#WORKFLOW FUNCTIONS

def get_selected_month_products(df, selected_month):
    """
    Given the complete dataframe and a selected month as a number, it returns the product purchases of all users for that month    
    """
    keep_cols = ['cod_persona','ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']

    df['mes_standard'] = pd.to_datetime(df["mes"])
    df['month_only'] = df['mes_standard'].apply(lambda row: row.month)
    df['year_only'] = df['mes_standard'].apply(lambda row: row.year)
    df_subset = df[(df['year_only']  == 2016) & (df['month_only'] == selected_month)] #Se filtra por el último año para solo tener una recomendación
    df_subset = df_subset[keep_cols]
    df_subset = df_subset.sort_values('cod_persona', ascending = True)
    df_subset.set_index('cod_persona', inplace = True) 

    return df_subset

def get_inverted_matrix(df):
    """
    Given a matrix of 1s and 0s, it returns a matrix with the values inverted
    """
    df_inverted = df.replace({0:1, 1:0})
    return df_inverted

def combined_raw_recommender_system(m1 ,m2, m3, mask, threshold = 0.5):
    m1 = raw_products_2_recommend(m1, mask)
    m2 = raw_products_2_recommend(m2, mask)
    m3 = raw_products_2_recommend(m3, mask)
    rec_total = (m1 + m2 + m3)/3 

    rec_total[rec_total >= threshold] = 1
    rec_total[rec_total < threshold] = 0
    rec_total = rec_total.astype(int)

    return rec_total  

def raw_products_2_recommend(all_recommendations, mask):
    return all_recommendations * mask


def ranking_products(df):
    df_product = df[df.columns[20:45]]
    product_total_purchase = pd.DataFrame(df_product.sum(axis=0),columns=['purchase_times'])
    product_total_purchase.reset_index(inplace=True)
    product_popularity = product_total_purchase.sort_values('purchase_times',ascending=False)
    product_popularity['ranking'] = range(1,26,1)
    product_popularity.drop('purchase_times',axis=1,inplace=True)
    ranking_array = np.array(product_popularity.sort_index().ranking)

    return ranking_array


def masked_argmin(a,limit): 
    valid_idx = np.where(a > limit)[0]
    return valid_idx[a[valid_idx].argmin()]


# MODEL EVALUATION

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    actual = str(actual)
    predicted = str(predicted)

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def apk_scores(df, month, df_recommendation_by_most_popular,
               df_recommendation_by_users,df_recommendation_by_products,df_final_recommendation):
    
    """This only evaluates does that are present in April"""
    
    list_score_mp = []
    list_score_bu = []
    list_score_bp = []
    list_score_intelligent = []

    y_mp = []
    y_bu = []
    y_bp = []
    y_intelligent = []
    y_real = []

    mth = month
    df_mes = get_selected_month_products(df, mth) 
    #users = df_mes.index.to_list() #If there is new users in that month this could give problems
    users = [user for user in df_mes.index.to_list() if user in df_recommendation_by_most_popular.index.to_list()]

    for u in users:
        y_real.append(df_mes.loc[u].to_list())
        y_mp.append(df_recommendation_by_most_popular.loc[u].to_list())
        y_bu.append(df_recommendation_by_users.loc[u].to_list())
        y_bp.append(df_recommendation_by_products.loc[u].to_list())
        y_intelligent.append(df_final_recommendation.loc[u].to_list())


    for i in range(len(y_real)):
        list_score_mp.append(mapk(actual = y_real[i], predicted = y_mp[i], k = 25))
        list_score_bu.append(mapk(actual = y_real[i], predicted = y_bu[i] , k = 25))
        list_score_bp.append(mapk(actual = y_real[i], predicted = y_bp[i], k = 25))
        list_score_intelligent.append(mapk(actual = y_real[i], predicted = y_intelligent[i] , k = 25))

    score_mp = np.mean(np.asarray(list_score_mp))
    score_bu = np.mean(np.asarray(list_score_bu))
    score_bp = np.mean(np.asarray(list_score_bp))
    score_intelligent = np.mean(np.asarray(list_score_intelligent))

 
    print(f'Most popular APK Score: {score_mp} \n')
    print(f'Based in user APK Score: {score_bu} \n')
    print(f'Based in product APK Score: {score_bp} \n')
    print(f'Intelligent APK Score: {score_intelligent} \n')

    #save_predictions(y_mp, y_bu, y_bp, y_intelligent)
    
def apk_scores2(df_mes, df_recommendation_by_most_popular, df_recommendation_by_users,
                df_recommendation_by_products,df_final_recommendation):
    
    list_score_mp = []
    list_score_bu = []
    list_score_bp = []
    list_score_intelligent = []

    y_mp = []
    y_bu = []
    y_bp = []
    y_intelligent = []
    y_real = []

    #users = df_mes.index.to_list() #If there is new users in that month this could give problems
    users = [user for user in df_mes.index.to_list() if user in df_recommendation_by_most_popular.index.to_list()]

    
    for u in users:
        y_real.append(df_mes.loc[u].to_list())
        y_mp.append(df_recommendation_by_most_popular.loc[u].to_list())
        y_bu.append(df_recommendation_by_users.loc[u].to_list())
        y_bp.append(df_recommendation_by_products.loc[u].to_list())
        y_intelligent.append(df_final_recommendation.loc[u].to_list())


    for i in range(len(y_real)):
        list_score_mp.append(mapk(actual = y_real[i], predicted = y_mp[i], k = 25))
        list_score_bu.append(mapk(actual = y_real[i], predicted = y_bu[i] , k = 25))
        list_score_bp.append(mapk(actual = y_real[i], predicted = y_bp[i], k = 25))
        list_score_intelligent.append(mapk(actual = y_real[i], predicted = y_intelligent[i] , k = 25))

    score_mp = np.mean(np.asarray(list_score_mp))
    score_bu = np.mean(np.asarray(list_score_bu))
    score_bp = np.mean(np.asarray(list_score_bp))
    score_intelligent = np.mean(np.asarray(list_score_intelligent))

 
    print(f'Most popular APK Score: {score_mp} \n')
    print(f'Based in user APK Score: {score_bu} \n')
    print(f'Based in product APK Score: {score_bp} \n')
    print(f'Intelligent APK Score: {score_intelligent} \n')

    #save_predictions(y_mp, y_bu, y_bp, y_intelligent)

    
def save_predictions( y_mp, y_bu, y_bp, y_intelligent):
    submission_path = '/content/drive/MyDrive/Colab Notebooks/MachineLearningProject/predictions/'

    df_predictions_mp = pd.DataFrame(y_mp, columns = [d], index = users_2_recommend)
    df_predictions_bu = pd.DataFrame(y_bu, columns = [d], index = users_2_recommend)
    df_predictions_bp = pd.DataFrame(y_bp, columns = [d], index = users_2_recommend)
    df_predictions_intelligent = pd.DataFrame(y_intelligent, columns = [d], index = users_2_recommend)

    df_predictions_mp.to_csv(submission_path + "most_popular_predictions.csv",header=True,index=True)
    df_predictions_bu.to_csv(submission_path + "based_user_predictions.csv",header=True,index=True)
    df_predictions_bp.to_csv(submission_path + "based_product_predictions.csv",header=True,index=True)
    df_predictions_intelligent.to_csv(submission_path + "intelligent_predictions.csv",header=True,index=True)
    print(f"all df predictions saved to {submission_path}")

"""## 1. Load preprocessed data <a id="1">
    
This is the data given by the professor:

### **Warning! Change your paths accordingly**
"""

path = "/Users/danielmarchan/Documents/MasterBigData/MachineLearning/Recommender_System/df_complete2.csv"

#Get the list of customers that have bought a product at least once 

df = pd.read_csv(path)

"""## 2. User based model <a id="2">

##### Here we will calculate the following: 
    - User-based similarity matrix
    - k-most similar users to each unique user
    - Predictions based on this model
"""

# USER INFO AGGREGATED
# Catch only the user info features sort them by mes and keep the last
df_userinfo = df.drop(['ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25'], axis = 1)

df_userinfo = df_userinfo.sort_values('mes', ascending = False)
df_userinfo_NT = df_userinfo.drop(['fecha1', 'mes'], axis = 1) 
df_userinfo_NT = df_userinfo_NT.drop_duplicates(subset = 'cod_persona', keep = 'first')

#Set cod_persona as the index
df_userinfo_NT.set_index('cod_persona', inplace = True)  

#NORMALIZE NUMERIC VARIABLES
sc = StandardScaler()
num_vars = df_userinfo_NT.select_dtypes(exclude=['object'])
df_userinfo_NNT = df_userinfo_NT
df_userinfo_NNT[num_vars.columns] = sc.fit_transform(num_vars)  

# ONE HOT ENCODING BUT INTELLIGENT
df_userinfo_NNT_dummies = make_dummies(df_userinfo_NNT, threshold = 0.01)

"""Before running the model, two objects need to be created:
- First, a list which contains all the ids of those individuals that have bought a product at least once during their time as clients
- Second, a dataframe which contains the individuals' purchase history where the entries are 1 if that person has bought a given product at least once and 0 otherwise

"""

#Create a list which contains all the unique users in the data that have bought a product at least once 

df_productinfo = df[['cod_persona','ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']]
df_productinfo["prod_total"] = df_productinfo.iloc[:,1:].sum(axis=1)
df_productinfo_Agg = df_productinfo.groupby('cod_persona').sum()
list_persona_ever_bought = list(df_productinfo_Agg[df_productinfo_Agg['prod_total'] > 0].index)

#Build a dataframe where any number of purchases of any product appear as 1 and 0 otherwise (Only for those individuals that have bought something at least once)
df_productinfo["prod_total"] = df_productinfo.iloc[:,1:].sum(axis=1)
df_productinfo_1s_0s  = df_productinfo.groupby('cod_persona').sum()
df_productinfo_1s_0s.drop('prod_total', axis = 1, inplace = True)
df_productinfo_1s_0s = df_productinfo_1s_0s.loc[list_persona_ever_bought]
df_productinfo_1s_0s = df_productinfo_1s_0s.apply(lambda x: [y if y == 0 else 1 for y in x])

"""Model

**Warning!** The modelling will take around one hour to produce the predictions
"""

#Build the similarity matrix
df_userinfo_NNT_similarity_matrix = get_cosine_similarity_matrix(df_userinfo_NNT_dummies)

#Make sure that all users are in the columns of the similarity matrix, but only the users that have bought something before must be included in the axis
only_bought_userinfo_NNT_similarity_matrix = df_userinfo_NNT_similarity_matrix.loc[list_persona_ever_bought]

#Find the k-most similar individuals to each individual in the data
users = only_bought_userinfo_NNT_similarity_matrix.columns
sim_users_dict = {}
for user in users:
    sim_users_dict[user] = only_bought_userinfo_NNT_similarity_matrix[[user]].sort_values(user, ascending = False)[1:251].index.to_list()
k_most_similar_users_df_by_user_info = pd.DataFrame(sim_users_dict)

#Build a dataframe with the users and the raw recommendations as probabilities
recommendations_index = ['ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']
recommendations = pd.DataFrame()
recommendations['index'] = recommendations_index
recommendations.set_index('index', inplace = True)

for user in users:
    recommendations[user] = list(df_productinfo_1s_0s.loc[list(k_most_similar_users_df_by_user_info[user])].sum(axis = 0)/len(k_most_similar_users_df_by_user_info))

recommendations = recommendations.transpose()

#Make the recommendation (1) if its probability passes a certain threshold
#After hyperparameter tunning, it was found that the best value of the threshold is 0.5
recommendations_0s_1s_by_user_info = recommendations.apply(lambda x: [0 if y < 0.5 else 1 for y in x]).astype(int)
#recommendations_0s_1s_by_user_info.to_csv("raw_recommendations_by_0s_1s.csv")

"""## 3. Product based model <a id="3">

##### Here we will calculate the following: 
    - Model-based similarity matrix
    - k-most similar users to each unique user
    - Predictions based on this model

Before running the model, one objects need to be created:
- A dataframe which contains only the purchasing history information of each client. This data is then standardized using a standard scaler. 


Note that the entries of this dataframe will be the total number of times that each client has bought a given product during their time as clients.
"""

#Build a dataframe only with the purchase history information of each client in the data 

df_productinfo_Agg.drop('prod_total', axis = 1, inplace = True)

#We standardize the purchase history data before calculating the similarity matrix
sc = StandardScaler()
data = sc.fit_transform(df_productinfo_Agg)
df_productinfo_AggN = pd.DataFrame(data, index = df_productinfo_Agg.index, columns = df_productinfo_Agg.columns)

"""Model

**Warning!** The modelling will take around one hour to produce the predictions
"""

#Build the similarity matrix
df_productinfo_AggN_similarity_matrix = get_cosine_similarity_matrix(df_productinfo_AggN)

#Make sure that all users are in the columns of the similarity matrix, but only the users that have bought something before must be included in the axis
only_productinfo_AggN_similarity_matrix = df_productinfo_AggN_similarity_matrix.loc[list_persona_ever_bought]
only_productinfo_AggN_similarity_matrix

#For each user in the column space of the similarity matrix, get their most similar peers based on the cosine similarity of their purchase patterns
users_by_product = only_productinfo_AggN_similarity_matrix.columns
sim_users_dict = {}
for user in users_by_product:
    sim_users_dict[user] = only_productinfo_AggN_similarity_matrix[[user]].sort_values(user, ascending = False)[1:251].index.to_list()
k_most_similar_users_df_by_product_info = pd.DataFrame(sim_users_dict)

#Build a dataframe with the users and the raw recommendations as probabilities
recommendations_index = ['ind_prod1', 'ind_prod2', 'ind_prod3',
       'ind_prod4', 'ind_prod5', 'ind_prod6', 'ind_prod7', 'ind_prod8',
       'ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 'ind_prod13',
       'ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
       'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23',
       'ind_prod24', 'ind_prod25']
recommendations_by_product = pd.DataFrame()
recommendations_by_product['index'] = recommendations_index
recommendations_by_product.set_index('index', inplace = True)

for user in users_by_product:
    recommendations_by_product[user] = list(df_productinfo_1s_0s.loc[list(k_most_similar_users_df_by_product_info[user])].sum(axis = 0)/len(k_most_similar_users_df_by_product_info))
recommendations_by_product = recommendations_by_product.transpose()

#Make the recommendation (1) if its probability passes a certain threshold
#After hyperparameter tunning, it was found that the best value of the threshold is 0.5
recommendations_by_product_0s_1s = recommendations_by_product.apply(lambda x: [0 if y < 0.5 else 1 for y in x]).astype(int)
#recommendations_by_product_0s_1s.to_csv("raw_recommendations_by_product_0s_1s.csv")

"""## 4. Most popular product model <a id="4">

The most popular product model depends on the month to predict so it is executed in the workflow. It is important to remark that the most_popular_prods is basically a numpy array in which each position of the array corresponds to the number of each product (For example, index 0 is product 1) and inside of each array cell there is the ranking number of each product.
"""

most_popular_prods = ranking_products(df)
most_popular_prods

"""## 5. *Workflow*<a id="5">

#### a) Read data

### **Warning! Change your paths accordingly**
"""

df = pd.read_csv('df_complete2.csv') #df is always the same

df_raw_recommendation_by_users = pd.read_csv('raw_recommendations_by_users_0s_1s.csv', index_col = 0)

df_raw_recommendation_by_products = pd.read_csv('raw_recommendations_by_product_0s_1s.csv', index_col = 0)

"""#### b) Pick the users in a given month of 2016"""

month = 3 #Only of 2016 (3==March to predict April) 
df_month_products = get_selected_month_products(df, month) #Catch the products you have bought this last month (March)
df_mask_products = get_inverted_matrix(df_month_products)
users_2_recommend = list(df['cod_persona'].sort_values().unique())  #df_mask_products.index.to_list()
users_2_mask = df_mask_products.index.to_list()

"""#### c)  Make recommendations"""

#20 mins to execute

d = ['ind_prod1', 'ind_prod2', 'ind_prod3','ind_prod4', 'ind_prod5', 'ind_prod6', 
     'ind_prod7', 'ind_prod8','ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 
     'ind_prod13','ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
     'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23','ind_prod24', 'ind_prod25']

# MOST COMMON RECOMMENDATION
# This model does not used the mask recommendations cause it has a more complex approach that 
# basically recommends in order based on the masked products
df_recommendation_by_most_popular = recommendations_by_most_popular(users_2_recommend, most_popular_prods, 
                                                                    df_mask_products,num_recommend = 1)

# CUSTOMER BASED RECOMMENDATION

df_recommendation_by_users = mask_recommendations(users_2_recommend, df_raw_recommendation_by_users, 
                                    df_mask_products)

# PRODUCT BASED RECOMMENDATION

df_recommendation_by_products = mask_recommendations(users_2_recommend, df_raw_recommendation_by_products, 
                                    df_mask_products)

# FINAL INTELLIGENT VOTING SYSTEM FROM THE 3 MODELS
    
final_recommendations = {}

mask = np.ones(25, dtype=int) #A mask was already pass to all the models before 

for user in users_2_recommend:
    m_prod_based = df_recommendation_by_products.loc[user].to_numpy()
    m_user_based = df_recommendation_by_users.loc[user].to_numpy()
    m_mostPop_based = df_recommendation_by_most_popular.loc[user].to_numpy()
    final_recommend = combined_raw_recommender_system(m_prod_based ,m_user_based, m_mostPop_based, 
                                                      mask, threshold = 0.5)
    final_recommendations[user] = final_recommend 


df_final_recommendation = pd.DataFrame.from_dict(final_recommendations, orient='index', columns=d)
df_final_recommendation = df_final_recommendation.rename_axis("cod_persona")

df_final_recommendation

"""## 6. *Evaluate models for April*<a id="6">

Here we will evaluate our 3 models for April´s predictions. In a similar manner this function could be used to evaluate May predictions. Two different ways of evaluation:

##### First: given a df like the one given at the beginning of the project
"""

#Month 4 (April)
month_2_recommend = 4


apk_scores(df, month_2_recommend, df_recommendation_by_most_popular,
               df_recommendation_by_users,df_recommendation_by_products, df_final_recommendation)

"""##### Second: given the predictions in a dataframe with cod_persona as index and prod_1, prod_2, ... etc as columns"""

# ------ Example of the input format --------
month_2_recommend = 4
df_month_to_eval = get_selected_month_products(df, month_2_recommed)  #Only the purchases for that month
# --------------------------------------------

apk_scores2(df_month_to_eval, df_recommendation_by_most_popular, df_recommendation_by_users,
                df_recommendation_by_products,df_final_recommendation)

"""## 7. *Save predictions for May*<a id="7">"""

#20 mins to execute
month_previous_2_prediction = 4 #Only of 2016 (3==March to predict April) change this to 4 at the END
most_popular_prods = ranking_products(df)
df_month_products = get_selected_month_products(df, month_previous_2_prediction) #Esto solo te coge lo que ha comprado
df_mask_products = get_inverted_matrix(df_month_products)
users_2_recommend = list(df['cod_persona'].sort_values().unique())  #df_mask_products.index.to_list()
users_2_mask = df_mask_products.index.to_list()


d = ['ind_prod1', 'ind_prod2', 'ind_prod3','ind_prod4', 'ind_prod5', 'ind_prod6', 
     'ind_prod7', 'ind_prod8','ind_prod9', 'ind_prod10', 'ind_prod11', 'ind_prod12', 
     'ind_prod13','ind_prod14', 'ind_prod15', 'ind_prod16', 'ind_prod17', 'ind_prod18',
     'ind_prod19', 'ind_prod20', 'ind_prod21', 'ind_prod22', 'ind_prod23','ind_prod24', 'ind_prod25']

# MOST COMMON RECOMMENDATION
# This model does not used the mask recommendations cause it has a more complex approach that 
# basically recommends in order based on the masked products
df_recommendation_by_most_popular = recommendations_by_most_popular(users_2_recommend, most_popular_prods, 
                                                                    df_mask_products,num_recommend = 1)

# CUSTOMER BASED RECOMMENDATION

df_recommendation_by_users = mask_recommendations(users_2_recommend, df_raw_recommendation_by_users, 
                                    df_mask_products)

# PRODUCT BASED RECOMMENDATION

df_recommendation_by_products = mask_recommendations(users_2_recommend, df_raw_recommendation_by_products, 
                                    df_mask_products)

# FINAL INTELLIGENT VOTING SYSTEM FROM THE 3 MODELS
    
final_recommendations = {}

mask = np.ones(25, dtype=int) #A mask was already pass to all the models before 
for user in users_2_recommend:
    m_prod_based = df_recommendation_by_products.loc[user].to_numpy()
    m_user_based = df_recommendation_by_users.loc[user].to_numpy()
    m_mostPop_based = df_recommendation_by_most_popular.loc[user].to_numpy()
    final_recommend = combined_raw_recommender_system(m_prod_based ,m_user_based, m_mostPop_based, 
                                                      mask, threshold = 0.5)
    final_recommendations[user] = final_recommend 


df_final_recommendation_may = pd.DataFrame.from_dict(final_recommendations, orient='index', columns=d)
df_final_recommendation_may = df_final_recommendation.rename_axis("cod_persona")

#df_final_recommendation_may.to_csv('df_final_recommendation.csv', header=True, index=True)

df_final_recommendation_may

df_final_recommendation_may.to_csv('df_final_recommendation_may.csv', header=True, index=True)

"""##### Evaluation: To evaluate this model change these cells to code mode an execute any of the two cells below depending on the data you want to use as input:

path = /path to the file you want to use as input

#First input option
df = pd.read(path)

#Second input option
df_month_to_eval = pd.read(path)

Given a df like the one given at the beginning of the project:

#Month 5 (May)

month_2_recommend = 5


apk_scores(df, month_2_recommend, df_recommendation_by_most_popular,
               df_recommendation_by_users,df_recommendation_by_products, df_final_recommendation)

Given the predictions in a dataframe with cod_persona as index and prod_1, prod_2, ... etc as columns

# ------ Example of the input format --------
#month_2_recommend = 5
#df_month_to_eval = get_selected_month_products(df, month_2_recommed)  #Only the purchases for that month
# --------------------------------------------

apk_scores2(df_month_to_eval, df_recommendation_by_most_popular, df_recommendation_by_users,
                df_recommendation_by_products,df_final_recommendation)
"""