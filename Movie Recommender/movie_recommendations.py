#!/usr/bin/env python

import matplotlib.pyplot as plt
import movie_data_helper
import numpy as np
import scipy
import scipy.special
from sys import exit


# Check Readme for details of the implementations and notations used in the equations
def compute_posterior(prior, likelihood, y):
    """
    Use Bayes' rule for random variables to compute the posterior distribution of a hidden variable X, given N
    observations Y_0, Y_1, ..., Y_{N-1}.Conditioned on X, these observations Y_0, Y_1, ..., Y_{N-1} are i.i.d.

    Hidden random variable X is assumed to take on a value in {0, 1, ..., M-1}.

    Each random variable Y_i takes on a value in {0, 1, ..., K-1}.

    Inputs
    ------
    - prior: a length M vector stored as a 1D NumPy array; prior[m] gives the (unconditional) probability that X = m
    - likelihood: a K row by M column matrix stored as a 2D NumPy array; likelihood[k, m] gives the probability that
      Y = k given X = m
    - y: a length-N vector stored as a 1D NumPy array; y[n] gives the observed value for random variable Y_n

    Output
    ------
    - posterior: a length M vector stored as a 1D NumPy array: posterior[m] gives the probability that X = m given
      Y_0 = y_0, ..., Y_{N-1} = y_{n-1}
    """

    # ERROR CHECKS

    # check that prior probabilities sum to 1
    if np.abs(1 - np.sum(prior)) > 1e-06:
        exit('In compute_posterior: The prior probabilities need to sum to 1')

    # check that likelihood is specified as a 2D array
    if len(likelihood.shape) != 2:
        exit('In compute_posterior: The likelihood needs to be specified as a 2D array')

    K, M = likelihood.shape

    # make sure likelihood and prior agree on number of hidden states
    if len(prior) != M:
        exit('In compute_posterior: Mismatch in number of hidden states according to the prior and the likelihood')

    # make sure the conditional distribution given each hidden state value sums
    # to 1
    for m in range(M):
        if np.abs(1 - np.sum(likelihood[:, m])) > 1e-06:
            exit('In compute_posterior: P(Y | X = %d) does not sum to 1' % m)

    # END OF ERROR CHECKS

    # Using logarithmic scale to avoid running to math underflow issues when multiplying a lot of small fractions
    # Compute the log of the posterior
    log_prior = np.log(prior)

    # compute the log of the full likelihood P(y0...yn-1 | x)
    log_likelihood = np.log(likelihood)
    total_log_likelihood = np.zeros((1, M))
    for obs_number in range(len(y)):
        total_log_likelihood += log_likelihood[y[obs_number], :]

    # Compute Bayes Rule
    log_answer = (log_prior + total_log_likelihood) - scipy.special.logsumexp(log_prior + total_log_likelihood)

    # Apply np.exp to get the original value back from log scale
    posterior = np.exp(log_answer)

    return posterior


def compute_movie_rating_likelihood(M):
    """
    Compute the rating likelihood probability distribution of Y given X whereY is an individual rating (takes on a value
    in {0, 1, ..., M-1}), and X is the hidden true/inherent rating of a movie also takes on a value in {0, 1, ..., M-1}
    so (MxM) size

    Output
    ------
    - likelihood: an M row by M column matrix stored as a 2D NumPy array; likelihood[k, m] gives the probability that
      Y = k given X = m
    """

    # define the size to begin with
    likelihood = np.zeros((M, M))

    # Remember to normalize the likelihood, so that each column is a probability distribution.
    for m in np.arange(M):
        for k in np.arange(M):
            if k == m:
                likelihood[k, m] = 2
            else:
                likelihood[k, m] = 1./np.abs(k - m)
        likelihood[:, m] /= np.sum(likelihood[:, m])

    return likelihood


def infer_true_movie_ratings(num_observations=-1):
    """
    For every movie, computes the posterior distribution and MAP estimate of the movie's true/inherent rating given the
    movie's observed ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to use per movie (the default value of -1
      indicates that all available ratings will be used).

    Output
    ------
    - posteriors: a 2D array consisting of the posterior distributions where the number of rows is the number of movies,
      and the number of columns is M, i.e., the number of possible ratings (remember ratings are 0, 1, ..., M-1);
      posteriors[i] gives a length M vector that is the posterior distribution of the true/inherent rating of the i-th
      movie given ratings for the i-th movie (where for each movie, the number of observations used is precisely what is
      specified by the input variable`num_observations`)
    - MAP_ratings: a 1D array with length given by the number of movies; MAP_ratings[i] gives the true/inherent rating
      with the highest posterior probability in the distribution `posteriors[i]`
    """

    M = 11  # all of our ratings are between 0 and 10
    prior = np.array([1.0 / M] * M)  # uniform distribution
    likelihood = compute_movie_rating_likelihood(M)

    # get the list of all movie IDs to process
    movie_id_list = movie_data_helper.get_movie_id_list()
    num_movies = len(movie_id_list)

    posteriors = np.zeros((num_movies, M))
    MAP_ratings = np.zeros(num_movies)
    for movie_id in movie_id_list:
        ratings = movie_data_helper.get_ratings(movie_id)
        if num_observations > 0:
            ratings = ratings[:num_observations]
        posteriors[movie_id] = compute_posterior(prior, likelihood, y=ratings)
        MAP_ratings = np.argmax(posteriors, axis=1)

    return posteriors, MAP_ratings


def compute_entropy(distribution):
    """
    Given a distribution, computes the Shannon entropy of the distribution in bits.

    Input
    -----
    - distribution: a 1D array of probabilities that sum to 1

    Output:
    - entropy: the Shannon entropy of the input distribution in bits
    """
    # ERROR CHECK

    if np.abs(1 - np.sum(distribution)) > 1e-6:
        exit('In compute_entropy: distribution should sum to 1.')

    # END OF ERROR CHECK

    nonzero_indices = (distribution > 1e-08)
    entropy = -np.sum(distribution[nonzero_indices] * np.log2(distribution[nonzero_indices]))

    return entropy


def compute_true_movie_rating_posterior_entropies(num_observations):
    """
    For every movie, computes the Shannon entropy (in bits) of the posterior distribution of the true/inherent rating of
    the movie given observed ratings.

    Input
    -----
    - num_observations: integer that specifies how many available ratings to use per movie (the default value of -1
      indicates that all available ratings will be used)

    Output
    ------
    - posterior_entropies: a 1D array; posterior_entropies[i] gives the Shannon entropy (in bits) of the posterior
      distribution of the true/inherent rating of the i-th movie given observed ratings (with number of observed ratings
      given by the input `num_observations`)
    """

    posteriors = infer_true_movie_ratings(num_observations)[0]
    num_movies = posteriors.shape[0]
    posterior_entropies = np.zeros(num_movies)
    for movie_id in range(num_movies):
        posterior_entropies[movie_id] = compute_entropy(posteriors[movie_id])

    return posterior_entropies


def main():

    # Some ERROR CHECKS for FUNCTIONS
    print("Posterior calculation (few observations)")
    prior = np.array([0.6, 0.4])
    likelihood = np.array([
        [0.7, 0.98],
        [0.3, 0.02],
    ])
    y = [0]*2 + [1]*1
    print("My answer:")
    print(compute_posterior(prior, likelihood, y))
    print("Expected answer:")
    print(np.array([[0.91986917, 0.08013083]]))

    print("---")
    print("Entropy of fair coin flip")
    distribution = np.array([0.5, 0.5])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(1.0)

    print("Entropy of coin flip where P(heads) = 0.25 and P(tails) = 0.75")
    distribution = np.array([0.25, 0.75])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(0.811278124459)

    print("Entropy of coin flip where P(heads) = 0.75 and P(tails) = 0.25")
    distribution = np.array([0.75, 0.25])
    print("My answer:")
    print(compute_entropy(distribution))
    print("Expected answer:")
    print(0.811278124459)

    # END OF ERROR CHECKS

    posteriors, MAP_ratings = infer_true_movie_ratings()
    ratings_by_id = list(zip(movie_data_helper.get_movie_id_list(), MAP_ratings))
    ratings_by_id.sort(key=lambda x: x[1], reverse=True)

    print("Best movies by MAP estimate:")
    for i in range(10):
        print(movie_data_helper.get_movie_name(ratings_by_id[i][0]), ratings_by_id[i][1])

    print("---")
    print("Worst movies:")
    for i in range(-10, 0, 1):
        print(movie_data_helper.get_movie_name(ratings_by_id[i][0]), ratings_by_id[i][1])

    prob_of_10_by_id = list(zip(movie_data_helper.get_movie_id_list(), posteriors[:, 10]))
    prob_of_10_by_id.sort(key=lambda x: x[1], reverse=True)
    print("---")
    print("Movies most likely to be a perfect 10:")
    for i in range(10):
        print(movie_data_helper.get_movie_name(prob_of_10_by_id[i][0]), prob_of_10_by_id[i][1])

    max_observations = 200
    entropy_plot = np.zeros(max_observations)
    for num_observations in range(1, max_observations + 1):
        entropy_plot[num_observations - 1] = \
            np.mean(compute_true_movie_rating_posterior_entropies(num_observations))
        print(num_observations, entropy_plot[num_observations - 1])
    plt.plot(range(1, max_observations + 1), entropy_plot)
    plt.show()


if __name__ == '__main__':
    main()
