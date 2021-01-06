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
collab_predictions_file = './data/collab_predictions.csv'
collab_ratings_file = './data/collab_ratings.csv'
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
final_predictions = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
collab_predictions_description = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
collab_ratings_description = pd.read_csv(collab_ratings_file, delimiter = ',', header=None)

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

    print("writing ", name)
    with open(name, 'w') as mat_writer:
        mat = [map(str, row) for row in mat]
        mat = [','.join(row) for row in mat]
        mat = '\n'.join(mat)
        mat_writer.write(mat)
    
def write_vector(vec, name):
    print("writing ", name)
    with open(name, 'w') as vec_writer:
        vec = '\n'.join(vec)
        vec_writer.write(vec)

def write_predictions(predictions, name):
    print("writing ", name)
    with open(name, 'w') as submission_writer:
        #Formates data
        predictions = [map(str, row) for row in predictions]
        predictions = [','.join(row) for row in predictions]
        predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
        #Writes it down
        submission_writer.write(predictions)

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

def calculate_similarity(a, b):
    return cosineSim(a, b)

def create_similarities(mat):
    sim_mat = np.ones((mat.shape[1], mat.shape[1]))
    mat = mat.values.T

    for i, row in enumerate(mat):
        for j in range(i + 1, len(mat)):
            sim = calculate_similarity(mat[i], mat[j])
            sim_mat[i][j] = sim
            sim_mat[j][i] = sim

    write_mat(sim_mat, sim_mat_file)

def create_neighbourhood(mat, threshold):
    neighbourhood = np.empty(mat.shape[0] + 1, dtype=object)
    neighbourhood[0] = "!"
    print(len(neighbourhood))
    mat = mat.values
    for i, row in enumerate(mat):
        list = ""
        for j, val in enumerate(row):
            val = float(val)
            if val > threshold and val < 1:
                list = list + str(j) + ","
        if len(list) == 0:
            list += ","
        neighbourhood[i + 1] = list[:-1] + "!"

    write_vector(neighbourhood, neighbourhood_file)

#create_similarities(centered_mat_description)
# print("creating neighbourhood")
# create_neighbourhood(sim_mat_description, 0.0)
# print("done")

def check_similarities(neighbours, movieIndex, similarites):

    similarites = similarites.values

    row = neighbours[movieIndex - 1]
    for neighbour in row:
        val = similarites[int(neighbour) - 1][movieIndex]
        print("neighbour: ", str(neighbour), " sim: ", val)

# check_similarities(neighbourhood_description, 1, sim_mat_description)


def predict_with_neighbours(user, movie, neighbours, ratings, similarities):


    similarities = similarities.values
    ratings = ratings.values

    totalSim = 0
    for neighbour in neighbours[movie - 1]:
        neighbour = int(neighbour) - 1
        if ratings[user][neighbour] != 0:
            totalSim += similarities[neighbour][movie]
    
    if totalSim == 0:
        return global_avg

    finalRating = 0
    for neighbour in neighbours[movie - 1]:
        neighbour = int(neighbour) - 1
        rating = ratings[user][neighbour]
        sim = similarities[neighbour][movie]
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
    new_ratings = np.zeros((ratings.shape[0] + 1, ratings.shape[1] + 1))
    predictions = predictions.values
    actual_predictions = actual_predictions.values
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        val =  actual_predictions[i][1]
        new_ratings[user][movie] = float(val)

    for i in range(1, ratings.shape[0]):
        for j in range(1, ratings.shape[1]):
            if ratings[i - 1][j] != 0 and new_ratings[i][j] == 0:
                new_ratings[i][j] = ratings[i - 1][j]
                
                
    return new_ratings

# predictions = predict_collaborative_filtering(movies_description, users_description, ratings_mat_description, predictions_description)
# write_predictions(predictions, collab_predictions_file)

# ratings = ratings_mat_description.values
# print(ratings[:0].sum())
# print(collab_predictions_description.values[1][1])
# newRatings = apply_predictions(ratings, predictions_description, collab_predictions_description)
# write_mat(newRatings, collab_ratings_file)

#####
##
## LATENT FACTORS
##
#####

def create_gender_averages(users, ratings):

    #Create an empty matrix which will be filled with the columns: movieId, AvgMaleRating and AvgFemaleRating
    genderAverageMatrix = np.zeros(ratings.shape[1], 3)

    for movieId, movie in enumerate(ratings.T, 1):
        totalMaleRating = 0
        totalFemaleRating = 0
        totalRatings = 0

        for userId, movieRating in enumerate(movie, 1):
            if movieRating != 0:
                if users[userId][1] == "M":
                    totalMaleRating += movieRating
                else:
                    totalFemaleRating += totalFemaleRating
                totalRatings += 1

        #Set current movieId row
        genderAverageMatrix[movieId - 1] = (movieId, totalMaleRating / totalRatings, totalFemaleRating / totalRatings )

    return genderAverageMatrix

def find_age_group(userAge):

    base = int(userAge / 10)
    return base * 10


def create_age_group_averages(users, ratings):

    #Create an empty matrix which will be filled with a column containing the movieId and 10 colums corresponding to the amount of age groups we set
    ageGroupAverageMatrix = np.zeros(ratings.shape[1], 11)

    for movieId, movie in enumerate(ratings.T, 1):
        totalRatings = 0

        for userId, movieRating in enumerate(movie, 1):
            if movieRating != 0:
                ageGroup = find_age_group(users[userId][2])
                ageGroupAverageMatrix[movieId - 1][ageGroup] += movieRating
                totalRatings += 1
        
        ageGroupAverageMatrix[movieId - 1] = ageGroupAverageMatrix[movieId - 1] / totalRatings

    
    return ageGroupAverageMatrix

    
def predict_latent_factors(ratings, predictions):

    predictions = predictions.values

    print(ratings.values.shape)

    Q, S, Vt = np.linalg.svd(ratings.values, full_matrices=False)

    Sigma = np.zeros((len(S), len(S)))
    for i, s in enumerate(S):
        Sigma[i,i] = s

    Pt = np.matmul(Sigma, Vt)
    # reconstructed = np.matmul(Q, Pt)
    # write_mat(reconstructed, "./data/recon.csv")
    Submission = np.zeros((len(predictions), 2))
    for i, p in enumerate(predictions):
      Submission[i,0] = i + 1
      Submission[i,1] = np.dot(Q[p[0]], Pt[p[1]])
    
    return Submission
    ## TO COMPLETE    
    
predictions = predict_latent_factors(collab_ratings_description, predictions_description)
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
# predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)

write_predictions(predictions, submission_file)
