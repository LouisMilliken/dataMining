import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'
ratings_mat_file = './data/ratings_mat.csv'
centered_mat_file = './data/centered_mat.csv'
sim_mat_file = './data/similarity_mat.csv'
neighbourhood_file = './data/neighbourhoods.csv'
# Read the data using pandas
print("loading files...")
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
ratings_mat_description = pd.read_csv(ratings_mat_file, delimiter = ',')        #USERS ARE THE ROWS
centered_mat_description = pd.read_csv(centered_mat_file, delimiter = ',')      #MOVIES ARE THE COLUMNS
sim_mat_description = pd.read_csv(sim_mat_file, delimiter = ',')
neighbourhood_description = pd.read_csv(neighbourhood_file, delimiter = '!', squeeze = 'true')

def valid(row, i):
    try:
        return row[0].split(",")
    except AttributeError:
        return [i + 1]


def unzip_neighbourhood():
    neighbourhood = np.empty(neighbourhood_description.shape[0], dtype=object)
    nd = neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

neighbourhood_description = unzip_neighbourhood()
print("done")
global_avg = 3.58131489029763
#####
##
## COLLABORATIVE FILTERING
##
#####

def write_mat(mat, name):

    with open(name, 'w') as mat_writer:
        mat = [map(str, row) for row in mat]
        mat = [','.join(row) for row in mat]
        mat = '\n'.join(mat)
        mat_writer.write(mat)
    
def write_vector(vec, name):
    with open(name, 'w') as vec_writer:
        vec = '\n'.join(vec)
        vec_writer.write(vec)


def create_centered_mat(ratings):
    ratings = ratings.values
    centered_mat = np.zeros((ratings.shape[0] + 1, ratings.shape[1] + 1))
    totalSum = 0
    totalNum = 0
    for y, row in enumerate(ratings):
        sum = 0
        num = 0
        len = row.size
        for val in row:
            if float(val) > 0:
                sum += float(val)
                num += 1
        totalSum += sum
        totalNum += num
        avg = sum / num
        #print(avg)
        for x, val in enumerate(row):
            if x >= len:
                break
            if float(val) > 0.0:
                centered_mat[y + 1][x] = int(val) - avg
    write_mat(centered_mat, centered_mat_file)

#create_centered_mat(ratings_mat_description)

def cosineSim(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_similarities(mat):
    sim_mat = np.ones((mat.shape[1], mat.shape[1]))
    mat = mat.values.T

    for i, row in enumerate(mat):
        for j in range(i + 1, len(mat)):
            sim = cosineSim(mat[i], mat[j])
            sim_mat[i][j] = sim
            sim_mat[j][i] = sim

    write_mat(sim_mat, sim_mat_file)

def create_neighbourhood(mat, threshold):
    neighbourhood = np.empty(mat.shape[0], dtype=object)
    mat = mat.values
    for i, row in enumerate(mat, ):
        list = ""
        for j, val in enumerate(row):
            
            if val > threshold and val < 1:
                list = list + str(j) + ","

        neighbourhood[i] = list[:-1] + "!"

    write_vector(neighbourhood, neighbourhood_file)

#create_similarities(centered_mat_description)
#create_neighbourhood(sim_mat_description, 0)



def predict_with_neighbours(user, movie, neighbours, ratings, similarities):


    similarities = similarities.values
    ratings = ratings.values

    totalSim = 0
    for neighbour in neighbours[movie]:
        neighbour = int(neighbour)
        if ratings[user][neighbour] != 0:
            totalSim += similarities[movie][neighbour]
    
    if totalSim == 0:
        return global_avg

    finalRating = 0
    for neighbour in neighbours[movie]:
        neighbour = int(neighbour)
        rating = ratings[user][neighbour]
        sim = similarities[movie][neighbour]
        if rating != 0:
            finalRating += rating * sim / totalSim
    
    return finalRating
    

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = i + 1
        finalPredictions[i][1] = predict_with_neighbours(user - 1, movie - 1, neighbourhood_description, ratings, sim_mat_description)
    
    return finalPredictions

def apply_predictions(ratings, predictions, actual_predictions):
    new_ratings = ratings.values
    predictions = predictions.values

    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        new_ratings[user, movie] = actual_predictions[i]
    return new_ratings

predictions = predict_collaborative_filtering(movies_description, users_description, ratings_mat_description, predictions_description)
#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
    # mat = predict_collaborative_filtering(movies, users, ratings, predictions)

    # Q, S, Vt = np.linalg.svd(mat)

    # Sigma = np.zeros(len(S), len(S))
    # for i, s in enumerate(S):
    #     Sigma[i,i] = s

    # Pt = np.dot(Sigma, Vt)
    
    # Submission = np.zeros((len(predictions), 2))
    # for i, p in enumerate(predictions):
    #   Submission[i,0] = i
    #   Submission[i,1] = np.dot(Q[p.userID], Pt[p.movieID])

    # return Submission
    
    
    ## TO COMPLETE

    pass
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
  ## TO COMPLETE

  pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]



   
def create_ratings_mat(movies, users, ratings):
    ratings_mat = np.zeros((len(users) + 1, len(movies) + 1))
    for i, row in enumerate(ratings.T.iteritems()):
        ratings_mat[row[1].userID, row[1].movieID] = row[1].rating

    write_mat(ratings_mat, ratings_mat_file)
    
    return ratings_mat



#####
##
## SAVE RESULTS
##
##### 
#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)