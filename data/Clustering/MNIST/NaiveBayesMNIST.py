import random
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt

def read_in_data(json_file):
	"""
	to parse the a line of the digits file into tuples of
	(labelled digit, numpy array of vector representation of digit)
	"""
	json_object = loads(json_file)
	json_data = b64decode(json_object["data"])
	digit_vector = np.fromstring(json_data, dtype=np.ubyte)
	digit_vector = digit_vector.astype(np.float64)
	return (json_object["label"], digit_vector)

# Digits is a list of 60,000 tuples,
# each containing a labelled digit and its vector representation.
with open("/home/cloudera/Desktop/digits.base64.json","r") as f:
	digits = map(read_in_data, f.readlines())


# pick a ratio for splitting the digits list
# into a training and a validation set.
training_size = int(len(digits)*0.25)
validation = digits[:training_size]
training = digits[training_size:]


def display_digit(digit, labeled = True, title = ""):
	"""
	graphically displays a 784x1 vector, representing a digit
	"""
	if labeled:
		digit = digit[1]
	image = digit
	plt.figure()
	fig = plt.imshow(image.reshape(28,28))
	fig.set_cmap('gray_r')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	if title != "":
		plt.title("Inferred label: " + str(title))


def init_centroids(labelled_data,k):
	"""
	randomly pick some k centers from the data as starting values
	for centroids. Remove labels.
	"""
	return map(lambda x: x[1], random.sample(labelled_data,k))

def sum_cluster(labelled_cluster):
	"""
	from http://stackoverflow.com/a/20642156
	element-wise sums a list of arrays.
	"""
	# assumes len(cluster) > 0
	sum_ = labelled_cluster[0][1].copy()
	for (label,vector) in labelled_cluster[1:]:
		sum_ += vector
	return sum_

def mean_cluster(labelled_cluster):
	"""
	compute the mean (i.e. centroid at the middle)
	of a list of vectors (a cluster):
	take the sum and then divide by the size of the cluster.
	"""
	sum_of_points = sum_cluster(labelled_cluster)
	mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
	return mean_of_points


def form_clusters(labelled_data, unlabelled_centroids):
	"""
	given some data and centroids for the data, allocate each
	datapoint to its closest centroid. This forms clusters.
	"""
	# enumerate because centroids are arrays which are unhashable
	centroids_indices = range(len(unlabelled_centroids))
	# initialize an empty list for each centroid. The list will
	# contain all the datapoints that are closer to that centroid
	# than to any other. That list is the cluster of that centroid.
	clusters = {c: [] for c in centroids_indices}
	for (label,Xi) in labelled_data:
		# for each datapoint, pick the closest centroid.
		smallest_distance = float("inf")
		for cj_index in centroids_indices:
			cj = unlabelled_centroids[cj_index]
			distance = np.linalg.norm(Xi - cj)
			if distance < smallest_distance:
				closest_centroid_index = cj_index
				smallest_distance = distance
		# allocate that datapoint to the cluster of that centroid.
		clusters[closest_centroid_index].append((label,Xi))
	return clusters.values()

def move_centroids(labelled_clusters):
	"""
	returns list of mean centroids corresponding to clusters.
	"""
	new_centroids = []
	for cluster in labelled_clusters:
		new_centroids.append(mean_cluster(cluster))
	return new_centroids


def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids):
	"""
	form clusters around centroids, then keep moving the centroids
	until the moves are no longer significant.
	"""
	previous_max_difference = 0
	while True:
		unlabelled_old_centroids = unlabelled_centroids
		unlabelled_centroids = move_centroids(labelled_clusters)
		labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
		# keep old_clusters and clusters so we can get the maximum difference
		# between centroid positions every time.
		differences = map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids)
		max_difference = max(differences)
		difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100
		previous_max_difference = max_difference
		# difference change is nan once the list of differences is all zeroes.
		if np.isnan(difference_change):
			break
	return labelled_clusters, unlabelled_centroids



def cluster(labelled_data, k):
	"""
	runs k-means clustering on the data.
	"""
	centroids = init_centroids(labelled_data, k)
	clusters = form_clusters(labelled_data, centroids)
	final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
	return final_clusters, final_centroids


def assign_labels_to_centroids(clusters, centroids):
	"""
	Assigns a digit label to each centroid. Note: This function
	 depends on clusters and centroids being in the same order.
	"""
	labelled_centroids = []
	for i in range(len(clusters)):
		labels = map(lambda x: x[0], clusters[i])
		# pick the most common label
		most_common = max(set(labels), key=labels.count)
		centroid = (most_common, centroids[i])
		labelled_centroids.append(centroid)
	return labelled_centroids

def classify_digit(digit, labelled_centroids):
	"""
	given an unlabelled digit represented by a vector and a list of
	labelled centroids [(label,vector)], determine closest centroid
	and thus classify the digit.
	"""
	mindistance = float("inf")
	for (label, centroid) in labelled_centroids:
		distance = np.linalg.norm(centroid - digit)
		if distance < mindistance:
			mindistance = distance
			closest_centroid_label = label
	return closest_centroid_label

def get_error_rate(labelled_digits,labelled_centroids):
	"""
	classifies a list of labelled digits. returns the error rate.
	"""
	classified_incorrect = 0
	for (label,digit) in labelled_digits:
		classified_label =classify_digit(digit, labelled_centroids)
		if classified_label != label:
			classified_incorrect +=1
	error_rate = classified_incorrect / float(len(digits))
	return error_rate

k = 16
clusters, centroids = cluster(training, k)
labelled_centroids = assign_labels_to_centroids(clusters, centroids)

for (label,digit) in labelled_centroids:
	display_digit(digit, labeled=False, title=label)

plt.show()


twos = []
frequency = {x:0 for x in range(10)}

for (label,digit) in validation:
	inferred_label = classify_digit(digit, labelled_centroids)
	if inferred_label==2:
		twos.append(digit)
		frequency[label] +=1

error_rates = {x:None for x in range(5,25)+[100]}
for k in range(5,25):
	clusters, centroids = cluster(training, k)
	label_centroids =assign_labels_to_centroids(clusters,centroids)
	error_rate = get_error_rate(validation, label_centroids)
	error_rates[k] = error_rate

# Show the error rates
x_axis = sorted(error_rates.keys())
y_axis = [error_rates[key] for key in x_axis]
plt.figure()
plt.title("Error Rate by Number of Clusters")
plt.scatter(x_axis, y_axis)
plt.xlabel("Number of Clusters")
plt.ylabel("Error Rate")
plt.show()