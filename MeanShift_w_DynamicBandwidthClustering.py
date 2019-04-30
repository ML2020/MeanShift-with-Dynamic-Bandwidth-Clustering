
# Hand coded Mean Shift Clustering code
# with Dynamic Bandwidth/Radii where we
# can vary our radius to determine how
# our input data points X are to be clustered
#
# Hand coded with a fixed radius only works best
# with a radius of about 4 for a given dataset
#
# but what if the dataset has 25 dimensions?
# we're going to add weighting inside of a 
# fixed radius, penalizing radii that are far
# away from our cluster centers
# 

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')

# make up some random sample data points
# utilize X and y since we might be doing clustering
# or Support Vectore Machines 
#
X, y = make_blobs(n_samples=25, centers=3, n_features=2)

#X = np.array([[1, 2],
#              [1.5, 1.8],
#              [5, 8],
#              [8, 8],
#              [1, 0.6],
#              [9, 11],
#              [8, 2],
#              [10,2],
#              [9, 3]])

# Note: because we have three generally
# cluster groups defined abobe, if we use
# our_radius < 3, some groups of data points
# will directly get their own overlapping
# cluster point
#
our_radius = 4

# scatter all the 0th and 1th elements in the array
#
#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]


# Mean Shift algo steps:
# 1. assign every single feature set is a cluster center
#
# 2. then take all of the data pts or feature sets within each 
#    cluster's radius (or within the bandwidth) and take the 
#    mean of all those datasets or feature sets: that is our 
#    new cluster center
#
# 3. then repeat step 2 until you have convergence which
#   many of the clusters converge on each orther or stop
#   moving
# 
 
# Setting radius_norm_step to 100 will add a lot
# of steps which will be individually weighted
# from center cluster(s).
#
# We want to have a lot of steps/bandwidths. 
# The closer they are to the centroid, the higher 
# the weight they will have
#

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
#find the center of all of our data pts
#
            all_data_centroid = np.average(data, axis=0)
    
# find the average distance of all data to the all data centroid
#
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        
# we now have a new self.radius from the entire dataset 
# norm divided by the norm step size
#

        centroids = {}
     
        for i in range(len(data)):
            centroids[i] = data[i]

# for Mean Shift you can have max_iterations and tolerance
# we'll have tolerance, but no max-iter
#
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

# now define out weighting factors
# this will loop for 0 through 99
# reversing the order of the list: 99, 98...
# Note: this is not that efficient re-defining
# the weight through each data point in X
# weights should really be defined above the 
# While loop
#
                weights = [i for i in range(self.radius_norm_step)][::-1]

# the weights will end up being the same for the centroids
# and through each iteration
#
                for featureset in data:
# if the Euclidean distance is less than the bandwidth we're 
# allowing for: we are within the bandwidth / radius, so 
# append the featureset data
#
# for dynamic bandwidth clustering, we're going
# to re-code the the following:
#
#                    if (np.linalg.norm(featureset-centroid) < self.radius):
#                        in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset-centroid)
# for when the feature set is comparing to itself
#
                    if distance == 0:
                        distance = 0.000000001

# now define out weight indices
# this is the entire distance divided by the 
# of radius steps we've taken
# i.e. the more we steps we've taken, the less 
# we want these steps to be
#
                    weight_index = int(distance / self.radius)

# if the weight_index is more than 100 steps 
# away, we just peg it to the max (99)
#
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

# this in theory is going to be a huge array or list
# of features, which need to average; one thing we can
# do is: here's 100 numbers to be meaned
# so this code is not as efficient as it could be
# as it's 100 times each feature possibly
#
                    to_add = (weights[weight_index] **2) * [featureset]

# add our raidu step
#
                    in_bandwidth += to_add  # a list plus a list
                
# the following gives us the mean vector 
# of all of our vectors
#
                new_centroid = np.average(in_bandwidth, axis=0)

# then add this to a new centroids list, and 
# append the tuple version of the centroid
# ie. we're converting an array to a numpy tuple
#
                new_centroids.append(tuple(new_centroid))
    
# Now we want to get the unique elements of the centroids
# we used a tuple above since we can use set() on tuples
# unique of each value on the array, sorting the list
#
# As we get convergence, we'll get identicals, which
# we don't need, sort the rest
#
            uniques = sorted(list(set(new_centroids)))
    
            to_pop = []

# there will be centroids close to each other, 
# but not exactly equal to each other, like tolerance
#
            for i in uniques:
                for ii in uniques:
                    if (i == ii):    # will already be equal to itself, so skip
                        pass

# if the distance between these two arrays/vectors
# less than one radius step of each other: within
# one radius step of each other, these need to be 
# converged to the same centroid:
#           
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break


# now we can remove the uniques that are
# within the step size
#
            for i in to_pop:
# may need a try: here:
                try:
                    uniques.remove(i)
                except:
                    pass
       
# our way of copying the centroids dict without
# taking the attributes, w/o modifying prev centroids
# 
            prev_centroids = dict(centroids)
    
# define a new empty centroids dictionary
#
            centroids = {}
            
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

# assume optimized until break
#
            optimized = True

# now we check for any centroids movement:
# if we've found one that's moved, there's
# no reason to continue
#

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

# if we've found one centroid that's moved, 
# we can break out
#
                if not optimized:
                    break

# break out of our while loop since if
# we are optimized:
#
            if optimized:
                break
        
#            if (i > (self.radius+1)):
#                    break

                    
# now we're optimized:
# where now outside the while True loop:
#
        self.centroids = centroids
    
    
# now let's add classifications and adding code
# to scale out to larger datasets
#
        self.classifications = {}

# we're going to classify on cluster 0, 1, 2 etc
#
        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]

# which ever centroid has the least distance:
# that's the classification index
#
            classification = distances.index(min(distances))

# classification of that feature set get's an
# an append of that featureset
#
            self.classifications[classification].append(featureset)

# here's our prediction function:
#
    def predict(self, data):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]

# which ever centroid has the least distance:
# that's the classification index
#
        classification = distances.index(min(distances))

# classification of that feature set get's an
# an append of that featureset
#
        return classification
            

# End of our MeanShift defn
#

clf = Mean_Shift()

# fit our X data defined abobe using our self-defined
# Mean_Shift algo
#
clf.fit(X)

centroids = clf.centroids

# now plot our feature set data and centroids
#

for classification in clf.classifications:
    color = colors[classification]
    
    i = 0;
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color = color, s=150, linewidths=5)
        print("idx=: ", i, "idxfeatureset[0]: ", featureset[0], "featureset[1]: ", featureset[1])
        i += 1

# plot our centroids:
#
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color ='k', marker='*', s=150)
    print("c=: ", c, "centroids[c][0]: ", centroids[c][0], "centroids[c][1]: ", centroids[c][1],)

plt.show()

    
##################  End ################

""" 
Output data:

idx=:  0 idxfeatureset[0]:  -8.72643371027 featureset[1]:  -4.86437760324
idx=:  1 idxfeatureset[0]:  -7.5859357847 featureset[1]:  -3.87286249805
idx=:  2 idxfeatureset[0]:  -7.21271375012 featureset[1]:  -4.70458849822
idx=:  3 idxfeatureset[0]:  -8.09714129144 featureset[1]:  -5.49204429021
idx=:  4 idxfeatureset[0]:  -6.87276201985 featureset[1]:  -4.23768160252
idx=:  5 idxfeatureset[0]:  -7.35901088371 featureset[1]:  -6.67558235223
idx=:  6 idxfeatureset[0]:  -8.04412206671 featureset[1]:  -5.93617836011
idx=:  7 idxfeatureset[0]:  -8.35223164234 featureset[1]:  -5.0859029472
idx=:  0 idxfeatureset[0]:  -3.48737886466 featureset[1]:  6.16069676605
idx=:  1 idxfeatureset[0]:  -3.70862468673 featureset[1]:  4.61018646302
idx=:  2 idxfeatureset[0]:  -4.16313979651 featureset[1]:  6.11489079587
idx=:  3 idxfeatureset[0]:  -2.59764869972 featureset[1]:  6.88672823706
idx=:  4 idxfeatureset[0]:  -4.11607827974 featureset[1]:  5.61758430738
idx=:  5 idxfeatureset[0]:  -5.62875919775 featureset[1]:  5.06003212836
idx=:  6 idxfeatureset[0]:  -4.21958280045 featureset[1]:  7.17465431328
idx=:  7 idxfeatureset[0]:  -3.06273960627 featureset[1]:  7.01749494734
idx=:  0 idxfeatureset[0]:  -3.8905935785 featureset[1]:  -0.179689650889
idx=:  1 idxfeatureset[0]:  -2.95606953391 featureset[1]:  1.47608574173
idx=:  2 idxfeatureset[0]:  -4.24160315625 featureset[1]:  -1.09194859947
idx=:  3 idxfeatureset[0]:  -3.60827211274 featureset[1]:  -0.440712377435
idx=:  4 idxfeatureset[0]:  -3.61874718776 featureset[1]:  -0.782115972894
idx=:  5 idxfeatureset[0]:  -2.14744622762 featureset[1]:  -0.0486735292985
idx=:  6 idxfeatureset[0]:  -2.69468020889 featureset[1]:  -1.45781715853
idx=:  7 idxfeatureset[0]:  -2.04348524584 featureset[1]:  -1.52186357388
idx=:  8 idxfeatureset[0]:  -3.7777175827 featureset[1]:  1.78226659176
c=:  0 centroids[c][0]:  -7.8450783619 centroids[c][1]:  -5.09803105822
c=:  1 centroids[c][0]:  -3.80472715991 centroids[c][1]:  6.14083301867
c=:  2 centroids[c][0]:  -3.29609460893 centroids[c][1]:  -0.434494239483

"""
