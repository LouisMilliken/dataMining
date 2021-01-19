from os import write
from typing import ValuesView
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
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
latent_ratings_file = "./data/latent_ratings.csv"
factorized_ratings_file = "./data/factorized_ratings.csv"
bad_neighbourhood_file = "./data/bad_neighbourhood.csv"
strict_neighbourhood_file = "./data/strict_neighbourhood.csv"
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
bad_neighbourhood_description = pd.read_csv(bad_neighbourhood_file, delimiter = '!', squeeze = 'true')
strict_neighbourhood_description = pd.read_csv(strict_neighbourhood_file, delimiter = ',')
final_predictions = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
collab_predictions_description = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
collab_ratings_description = pd.read_csv(collab_ratings_file, delimiter = ',', header=None)
latent_ratings_description = pd.read_csv(latent_ratings_file, delimiter = ',', header=None)
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


def unzip_bad_neighbourhood():
    neighbourhood = np.empty(bad_neighbourhood_description.shape[0], dtype=object)
    nd = bad_neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

neighbourhood_description = unzip_neighbourhood()
bad_neighbourhood_description = unzip_bad_neighbourhood()
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

def create_centered_movies_mat(ratings):
    ratings = ratings.values.T
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
    write_mat(centered_mat.T, centered_mat_file)

#create_centered_mat(ratings_mat_description)


def create_centered_users_mat(ratings):
    ratings = ratings.values
    centered_mat = np.zeros((ratings.shape[0] + 1, ratings.shape[1] + 1))
    for y, row in enumerate(ratings):
        sum = 0
        num = 0
        len = row.size
        for val in row:
            if float(val) > 0:
                sum += float(val)
                num += 1

        avg = sum / num
        #print(avg)
        for x, val in enumerate(row):
            if x >= len:
                break
            if float(val) > 0.0:
                centered_mat[y + 1][x] = int(val) - avg
    write_mat(centered_mat, centered_mat_file)

# def pearsonSimAge(userIdA, userIdB, ratings):


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
    neighbourhood[0] = ""
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

def create_strict_neighbourhood(mat, size):
    neighbourhood = np.zeros((mat.shape[0] + 1, size))

    print(len(neighbourhood))
    mat = mat.values
    for i, row in enumerate(mat):
        row = np.argsort(row[1:3707])
        row = row[-(size + 1):-1]
        neighbourhood[i] = row
        
    write_mat(neighbourhood, strict_neighbourhood_file)


def create_bad_neighbourhood(mat, threshold = -1):
    neighbourhood = np.empty(mat.shape[0] + 1, dtype=object)
    neighbourhood[0] = "!"
    print(len(neighbourhood))
    mat = mat.values
    for i, row in enumerate(mat):
        list = ""
        for j, val in enumerate(row):
            val = float(val)
            if threshold > val and val < 0:
                list = list + str(j) + ","
        if len(list) == 0:
            list += ","
        neighbourhood[i + 1] = list[:-1] + "!"

    write_vector(neighbourhood, bad_neighbourhood_file)

#create_similarities(centered_mat_description)
# print("creating strict neighbourhood")
# create_strict_neighbourhood(sim_mat_description, 300)
# print("done")

# print("creating bad neighbourhood")
# create_bad_neighbourhood(sim_mat_description, 100)
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
    
def predict_with_both_neighbours(user, movie, neighbours, ratings, similarities):

    similarities = similarities.values
    ratings = ratings.values

    totalSim = 0
    for neighbour in neighbours[movie - 1]:
        neighbour = int(neighbour) - 1
        if ratings[user][neighbour] != 0:
            totalSim += np.absolute(similarities[neighbour][movie])
    
    if totalSim == 0:
        return global_avg

    finalRating = 0
    for neighbour in neighbours[movie - 1]:
        neighbour = int(neighbour) - 1
        rating = ratings[user][neighbour]
        sim = similarities[neighbour][movie]
        if sim > 0 and rating != 0:
            finalRating += rating * sim / totalSim
        if sim < 0 and rating != 0:
            rating = 6 - rating   #if rating = 1, new rating = 6 - 1 = 5
            sim = np.absolute(sim)   #if sim = -1, new sim = 1
            finalRating += rating * sim / totalSim
    
    return finalRating

def predict_collaborative_filtering(movies, users, ratings, predictions, sim_mat, neighbourhood):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = i + 1
        finalPredictions[i][1] = predict_with_both_neighbours(user - 1, movie - 1, neighbourhood, ratings, sim_mat)
    
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

predictions = predict_collaborative_filtering(movies_description, users_description, ratings_mat_description, predictions_description,
 sim_mat_description, bad_neighbourhood_description)

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
    reconstructed = latent_ratings_description.values
    Submission = np.zeros((len(predictions), 2))
    for i, p in enumerate(predictions):
      Submission[i,0] = int(i + 1)
      Submission[i,1] = reconstructed[p[0]][p[1]]
    
    return Submission
    ## TO COMPLETE    
    

def factorize(ratings, P, Q, numFeatures, errorMeasure = 0.001 ,iterations = 100, lr = 0.002, regularization = 0.2):
    
    Q = Q.T
    # Each step we try to find a more optimal P and Q by updating row by row from the ratings matrix
    for step in range(iterations):
        if step % (100 / iterations) == 0:
            print(step, "/", (iterations))
        num = len(ratings[0])
        for i in range(num):
            if i % (100 / num) == 0:
                print(step, "/", (num))
            for j in range(len(ratings)):

                # If there is a rating try to approach it
                if ratings[j][i] > 0:
                    
                    # Calculate the current error at this particular rating
                    error = ratings[j][i] - np.matmul(P[j,:], Q[:,i])
                    # print("---")
                    # print(ratings[j][i], " - ", np.matmul(P[j,:], Q[:,i]))

                    error = error * 2

                    # print(error)
                    #print("---")

                    for feature in range(numFeatures):

                        pReg = regularization * P[j][feature]
                        qReg = regularization * Q[feature][i]
                        
                        qErr = error * Q[feature][i]
                        pErr = error * P[j][feature]

                        pGrad = (pErr) - (pReg)
                        qGrad = (qErr) - (qReg)

                        P[j][feature] += lr * pGrad
                        Q[feature][i] += lr * qGrad
        
        
        # Using matmul create the new ratings matrix and intilize its error to 0
        newR = np.matmul(P, Q)
        error = 0


        # Calculate the error rating using root mean square error
        for i in range(len(ratings[0])):
            for j in range(len(ratings)):

                # Only calculate if the rating is non zero
                if ratings[j][i] > 0:
                    error += pow(ratings[j][i] - np.dot(P[j,:], Q[:,i]), 2)
                    
                    for feature in range(numFeatures):
                        error+= (regularization / 2) * pow(P[j][feature], 2) + pow(Q[feature][i], 2)

        # Stop optimizing the matrices P and Q if the error is below a certain threshold                
        if error < errorMeasure:
            break

    return P, Q.T

def predict_with_factorize(ratings, numFeatures, predictions):
    
    shape = ratings.shape

    # Initialize the defactorized matrices P and Q with ones
    P = np.random.rand(shape[0] + 1, numFeatures)
    Q = np.random.rand(shape[1] + 1, numFeatures)

    # Using gradient descent calculate the optimal matrices P and Q
    P, Q = factorize(ratings, P, Q, numFeatures, iterations = 10)

    # Create and write the new ratings matrix using the factorized matrices P and Q 
    factorized_ratings = np.matmul(P, Q.T)
    write_mat(factorized_ratings, factorized_ratings_file)

    # Create empty submission matrix to fill with predicted ratings
    Submission = np.zeros((len(predictions), 2))
    
    # Loop over the predictions and get the new ratings from the factorized ratings matrix
    for i, p in enumerate(predictions):
        Submission[i,0] = int(i + 1)
        Submission[i,1] = factorized_ratings[p[0]][p[1]]

    return Submission
    

# predictions = predict_with_factorize(collab_ratings_description.values, 2, predictions_description.values)


def predict_final(movies, users, ratings, predictions):
  ## TO COMPLETE

  pass
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]
   
def predict_average(predictions):
    number_predictions = len(predictions)

    return [[idx, global_avg] for idx in range(1, number_predictions + 1)]

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
predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
write_predictions(predictions, './data/predict_random.csv')

predictions = predict_average(predictions_description)
write_predictions(predictions, './data/predict_average.csv')

