from os import write
from typing import ValuesView
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
latent_ratings_file = "./data/latent_ratings.csv"
factorized_ratings_file = "./data/factorized_ratings.csv"
centered_user_mat_file = "./data/centered_user_mat.csv"
user_avg_mat_file = "./data/user_avg_mat.csv"
pearson_sim_age_mat_file = "./data/pearson_sim_age_mat.csv"
pearson_sim_sex_age_mat_file = "./data/pearson_sim_sex_age_mat.csv"
pearson_neighbourhood_file = './data/pearson_neighbourhoods.csv'
pearson_sex_age_neighbourhood_file = './data/pearson_sex_age_neighbourhoods.csv'
pearson_age_predictions_file = './data/pearson_age_predictions.csv'
pearson_sex_age_predictions_file = './data/pearson_sex_age_predictions.csv'
cosine_sim_age_mat_file = "./data/cosine_sim_age_mat.csv"
cosine_sim_sex_age_mat_file = "./data/cosine_sim_sex_age_mat.csv"
cosine_age_neighbourhood_file = './data/cosine_age_neighbourhood.csv'
cosine_sex_age_neighbourhood_file = './data/cosine_sex_age_neighbourhood.csv'
cosine_age_predictions_file = './data/cosine_age_predictions.csv'
cosine_sex_age_predictions_file = './data/cosine_sex_age_predictions.csv'
# Read the data using pandas
print("loading files...")
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
ratings_mat_description = pd.read_csv(ratings_mat_file, delimiter = ',')        #USERS ARE THE ROWS
centered_mat_description = pd.read_csv(centered_mat_file, delimiter = ',')      #MOVIES ARE THE COLUMNS
sim_mat_description = pd.read_csv(sim_mat_file, delimiter = ',')
pearson_age_sim_mat_description = pd.read_csv(pearson_sim_age_mat_file, delimiter = ',')
pearson_sex_age_sim_mat_description = pd.read_csv(pearson_sim_sex_age_mat_file, delimiter = ',')

centered_user_mat_description = pd.read_csv(centered_user_mat_file, delimiter = ',')
neighbourhood_description = pd.read_csv(neighbourhood_file, delimiter = '!', squeeze = 'true')
pearson_age_mat_description = pd.read_csv(pearson_sim_age_mat_file, delimiter = ',')
pearson_sex_age_mat_description = pd.read_csv(pearson_sim_sex_age_mat_file, delimiter = ',')
pearson_neighbourhood_description = pd.read_csv(pearson_neighbourhood_file, delimiter = '!', squeeze = 'true')
pearson_sex_age_neighbourhood_description = pd.read_csv(pearson_sex_age_neighbourhood_file, delimiter = '!', squeeze = 'true')


cosine_age_sim_mat_description = pd.read_csv(cosine_sim_age_mat_file, delimiter = ',')
cosine_sex_age_sim_mat_description = pd.read_csv(cosine_sim_sex_age_mat_file, delimiter = ',')

cosine_age_neighbourhood_description = pd.read_csv(cosine_age_neighbourhood_file, delimiter = '!', squeeze = 'true')
cosine_sex_age_neighbourhood_description = pd.read_csv(cosine_sex_age_neighbourhood_file, delimiter = '!', squeeze = 'true')

cosine_age_predictions_description = pd.read_csv(cosine_age_predictions_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
cosine_sex_age_predictions_description = pd.read_csv(cosine_sex_age_predictions_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})


# final_predictions = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
# collab_predictions_description = pd.read_csv(submission_file, delimiter=',', names=['Id', 'Rating'], skiprows = 1, dtype={'Id':'int', 'Rating':'float64'})
# collab_ratings_description = pd.read_csv(collab_ratings_file, delimiter = ',', header=None)
# latent_ratings_description = pd.read_csv(latent_ratings_file, delimiter = ',', header=None)
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

def unzip_pearson_neighbourhood():
    neighbourhood = np.empty(pearson_neighbourhood_description.shape[0], dtype=object)
    nd = pearson_neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

def unzip_pearson_sex_age_neighbourhood():
    neighbourhood = np.empty(pearson_sex_age_neighbourhood_description.shape[0], dtype=object)
    nd = pearson_sex_age_neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

def unzip_cosine_age_neighbourhood():
    neighbourhood = np.empty(cosine_age_neighbourhood_description.shape[0], dtype=object)
    nd = cosine_age_neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

def unzip_cosine_sex_age_neighbourhood():
    neighbourhood = np.empty(cosine_sex_age_neighbourhood_description.shape[0], dtype=object)
    nd = cosine_sex_age_neighbourhood_description.values
    for i, row in enumerate(nd):
        
        neighbours = valid(row, i)
        neighbourhood[i] = neighbours
    return neighbourhood

neighbourhood_description = unzip_neighbourhood()
pearson_neighbourhood_description = unzip_pearson_neighbourhood()
pearson_sex_age_neighbourhood_description = unzip_pearson_sex_age_neighbourhood()

cosine_age_neighbourhood_description = unzip_cosine_age_neighbourhood()
cosine_sex_age_neighbourhood_description = unzip_cosine_sex_age_neighbourhood()
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

def create_centered_movie_mat(ratings):
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
        if num > 0 :
            avg = sum / num
            #print(avg)
            for x, val in enumerate(row):
                if x >= len:
                    break
                if float(val) > 0.0:
                    centered_mat[y + 1][x] = int(val) - avg
    print(totalSum/totalNum)
    write_mat(centered_mat.T, centered_mat_file)

def create_centered_user_mat(ratings):
    ratings = ratings.values
    centered_mat = np.zeros((ratings.shape[0] + 1, ratings.shape[1] + 1))
    user_avg_mat = np.zeros((ratings.shape[0] + 1, 2))
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
        if num > 0 :
            avg = sum / num
            user_avg_mat[y+1][0] = y+1
            user_avg_mat[y+1][1] = avg
            #print(avg)
            for x, val in enumerate(row):
                if x >= len:
                    break
                if float(val) > 0.0:
                    centered_mat[y + 1][x] = int(val) - avg
    write_mat(centered_mat, centered_user_mat_file)
    write_mat(user_avg_mat, user_avg_mat_file)

# create_centered_user_mat(ratings_mat_description)



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

def create_neighbourhood(mat, threshold, file):
    neighbourhood = np.empty(mat.shape[0] + 1, dtype=object)
    neighbourhood[0] = "!"
    print(len(neighbourhood))
    mat = mat.values
    for i, row in enumerate(mat):
        list = ""
        for j, val in enumerate(row):
            val = float(val)
            if val > threshold and val < 1:
                list = list + str(j+1) + ","
        if len(list) == 0:
            print("empty "  ,str(j))
            list += ","
        neighbourhood[i + 1] = list[:-1] + "!"

    write_vector(neighbourhood, file)


def create_pearson_age_similarites(mat, users):
    sim_mat = np.ones((mat.shape[0], mat.shape[0]))
    mat = mat.values
    users = users.values

    for i, row in enumerate(mat):
       
        sdA = np.std(mat[i])
        ageA = users[i][2]
        sexA = users[i][1]

        for j in range(i + 1, len(mat)):
            sdB = np.std(mat[j])
            ageB = users[j][2]
            sexB = users[j][1]

            ageFactor = 1
            diff = np.absolute(ageA - ageB)
            if ageA != 1 and ageB != 1 and diff != 0:
                ageFactor = np.log10(diff + 1)

            sim = 0
            if sdA != 0 and sdB != 0:
                dot = np.dot(mat[i], mat[j]) / 6040
                sim = dot / (sdA * sdB * ageFactor)

            sim_mat[i][j] = sim 
            sim_mat[j][i] = sim

    write_mat(sim_mat, pearson_sim_age_mat_file)

def create_cosine_age_similarites(mat, users):
    sim_mat = np.ones((mat.shape[0], mat.shape[0]))
    mat = mat.values
    users = users.values

    for i, row in enumerate(mat):
       
        ageA = users[i][2]

        for j in range(i + 1, len(mat)):
            ageB = users[j][2]

            ageFactor = 1
            diff = np.absolute(ageA - ageB)
            if ageA != 1 and ageB != 1 and diff != 0:
                ageFactor = np.log10(diff + 1)

            sim = cosineSim(mat[i], mat[j]) / ageFactor
            if sim > 1:
                sim = 1
            if sim < -1:
                sim = -1

            sim_mat[i][j] = sim 
            sim_mat[j][i] = sim

    write_mat(sim_mat, cosine_sim_age_mat_file)


def create_cosine_sex_age_similarites(mat, users, sexDiff = 0.05):
    sim_mat = np.ones((mat.shape[0], mat.shape[0]))
    mat = mat.values
    users = users.values

    for i, row in enumerate(mat):
       
        ageA = users[i][2]
        sexA = users[i][1]

        for j in range(i + 1, len(mat)):
            ageB = users[j][2]
            sexB = users[j][1]

            ageFactor = 1
            diff = np.absolute(ageA - ageB)
            if ageA != 1 and ageB != 1 and diff != 0:
                ageFactor = np.log10(diff + 1)

            sexFactor = 1
            if sexA != sexB:
                sexFactor += sexDiff
            else:
                sexFactor -= sexDiff

            sim = cosineSim(mat[i], mat[j]) / (ageFactor * sexFactor)
            if sim > 1:
                sim = 1
            if sim < -1:
                sim = -1

            sim_mat[i][j] = sim 
            sim_mat[j][i] = sim

    write_mat(sim_mat, cosine_sim_sex_age_mat_file)


def create_pearson_sex_age_similarites(mat, users, sexDiff = 0.05):
    sim_mat = np.ones((mat.shape[0], mat.shape[0]))
    mat = mat.values
    users = users.values

    for i, row in enumerate(mat):
       
        sdA = np.std(mat[i])
        ageA = users[i][2]
        sexA = users[i][1]

        for j in range(i + 1, len(mat)):
            sdB = np.std(mat[j])
            ageB = users[j][2]
            sexB = users[j][1]

            ageFactor = 1
            diff = np.absolute(ageA - ageB)
            if ageA != 1 and ageB != 1 and diff != 0:
                ageFactor = np.log10(diff + 1)
            
            sexFactor = 1
            if sexA != sexB:
                sexFactor -= sexDiff
            else:
                sexFactor += sexDiff

            sim = 0
            if sdA != 0 and sdB != 0:
                dot = np.dot(mat[i], mat[j]) / 6040
                sim = dot / (sdA * sdB * ageFactor * sexFactor)

            
            sim_mat[i][j] = sim 
            sim_mat[j][i] = sim

    write_mat(sim_mat, pearson_sim_sex_age_mat_file)

# create_cosine_age_similarites(centered_user_mat_description, users_description)
# create_cosine_sex_age_similarites(centered_user_mat_description, users_description)
# create_pearson_age_similarites(centered_user_mat_description, users_description)
# create_pearson_sex_age_similarites(centered_user_mat_description, users_description)
# create_pearson_sex_age_similarites(center)
# create_similarities(centered_mat_description)
# print("creating age neighbourhood")
# create_neighbourhood(pearson_age_mat_description, 0.0, cosine_age_neighbourhood_file)
# print("done")

# print("creating age neighbourhood")
# create_neighbourhood(pearson_age_mat_description, 0.0, cosine_sex_age_neighbourhood_file)
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
    
    if finalRating < 0 or finalRating > 5.0:
        return global_avg

    return finalRating

def predict_with_pearson_neighbours(user, movie, neighbours, ratings, similarities):


    similarities = similarities.values
    ratings = ratings.values
    # neighbours = neighbours.values
    
    totalSim = 0
    for neighbour in neighbours[user - 1]:
        neighbour = int(neighbour) - 1
        if ratings[neighbour][movie] != 0:
            totalSim += similarities[user-1][neighbour]
    
    if totalSim == 0:
        return global_avg

    finalRating = 0
    for neighbour in neighbours[user - 1]:
        neighbour = int(neighbour) - 1
        rating = ratings[neighbour][movie]
        sim = similarities[user-1][neighbour]
        if rating != 0:
            finalRating += rating * sim / totalSim
    
    if finalRating < 0 or finalRating > 5.0:
        return global_avg

    return finalRating
    

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = int(i + 1)
        finalPredictions[i][1] = predict_with_neighbours(user - 1, movie - 1, neighbourhood_description, ratings, sim_mat_description)
    
    return finalPredictions

def predict_pearson(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = int(i + 1)
        finalPredictions[i][1] = predict_with_pearson_neighbours(user - 1, movie - 1, pearson_neighbourhood_description, ratings, pearson_age_sim_mat_description)
    
    return finalPredictions
def predict_colab_age(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = int(i + 1)
        finalPredictions[i][1] = predict_with_pearson_neighbours(user - 1, movie - 1, cosine_age_neighbourhood_description, ratings, cosine_age_sim_mat_description)
    
    return finalPredictions

def predict_colab_sex_age(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = int(i + 1)
        finalPredictions[i][1] = predict_with_pearson_neighbours(user - 1, movie - 1, cosine_sex_age_neighbourhood_description, ratings, cosine_sex_age_sim_mat_description)
    
    return finalPredictions

def predict_sex_age_pearson(movies, users, ratings, predictions):
    # TO COMPLETE
    predictions = predictions.values
    finalPredictions = np.empty((len(predictions), 2))
    for i, prediction in enumerate(predictions):
        user = prediction[0]
        movie = prediction[1]
        finalPredictions[i][0] = int(i + 1)
        finalPredictions[i][1] = predict_with_pearson_neighbours(user - 1, movie - 1, pearson_sex_age_neighbourhood_description, ratings, pearson_sex_age_sim_mat_description)
    
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

# print(neighbourhood_description.shape)
# print("predicting.....")
# print(neighbourhood_description[0])
# print(cosine_age_neighbourhood_description[0])
# print(neighbourhood_description.shape)
# print(cosine_age_neighbourhood_description.shape)
# colab_age_predictions = predict_colab_age(movies_description, users_description, ratings_mat_description, predictions_description)
# write_predictions(colab_age_predictions, cosine_age_predictions_file)
# print("Half way there....")
# colab_sex_age_predictions = predict_colab_sex_age(movies_description, users_description, ratings_mat_description, predictions_description)
# write_predictions(colab_sex_age_predictions, cosine_sex_age_predictions_file)
# print("done!")
# print(pearson_age_sim_mat_description.values.shape)
# pearson_age_predictions = predict_pearson(movies_description, users_description, ratings_mat_description, predictions_description)
# write_predictions(pearson_age_predictions, pearson_age_predictions_file)

# pearson_sex_age_predictions = predict_sex_age_pearson(movies_description, users_description, ratings_mat_description, predictions_description)
# predictions = predict_collaborative_filtering(movies_description, users_description, ratings_mat_description, predictions_description)
# write_predictions(predictions, collab_predictions_file)

ratings = ratings_mat_description.values
# print(ratings[:0].sum())
# print(collab_predictions_description.values[1][1])
colab_age_ratings = apply_predictions(ratings, predictions_description, cosine_age_predictions_description)
colab_sex_age_ratings = apply_predictions(ratings, predictions_description, cosine_sex_age_predictions_description)
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
        for i in range(len(ratings[0])):
            for j in range(len(ratings)):

                # If there is a rating try to approach it
                if ratings[j][i] > 0:
                    
                    # Calculate the current error at this particular rating
                    error = ratings[j][i] - np.dot(P[j,:], Q[:,i])

                    
                    for feature in range(numFeatures):
                        pGrad = (2 * error * Q[feature][i]) - (regularization * P[j][feature])
                        qGrad = (2 * error * P[j][feature]) - (regularization * Q[feature][i])

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
                        e+= (regularization / 2) * pow(P[j][feature], 2) + pow(Q[feature][i], 2)

        # Stop optimizing the matrices P and Q if the error is below a certain threshold                
        if error < errorMeasure:
            break

    return P, Q.T

def predict_with_factorize(ratings, numFeatures, predictions):
    
    shape = ratings.shape

    # Initialize the defactorized matrices P and Q with ones
    P = np.ones((shape[0] + 1, numFeatures))
    Q = np.ones((shape[1] + 1, numFeatures))

    # Using gradient descent calculate the optimal matrices P and Q
    P, Q = factorize(ratings, P, Q, numFeatures)

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
    

# prediction = predict_with_factorize(collab_ratings_description.values, 1000, predictions_description.values)
colab_age_factorize_predictions = predict_with_factorize(colab_age_ratings, 15, predictions_description.values)
write_predictions(colab_age_factorize_predictions, './data/colab_age_factorize_predictions.csv')

colab_sex_age_factorize_predictions = predict_with_factorize(colab_sex_age_ratings, 15, predictions_description.values)
write_predictions(colab_sex_age_factorize_predictions, './data/colab_sex_age_factorize_predictions.csv')


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
# write_predictions(predictions, './data/submission.csv')
