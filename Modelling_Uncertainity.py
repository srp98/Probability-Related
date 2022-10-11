# Import Required Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Finite Probability Space
def sample_from_finite_probability_space(finite_prob_space):
    """
    Produces a random outcome from a given finite probability space
    :param finite_prob_space: finite probability space encoded as a dictionary
    :return: Random outcome, which is one of the keys in finite_prob_space dictionary's set of keys
    """
    # Produce a list of pairs of the form (outcome, outcome_probability)
    outcome_prob_pairs = list(finite_prob_space.items())

    # Convert the pairs into two lists of outcomes and outcome_probabilities
    outcomes, outcome_probs = zip(*outcome_prob_pairs)

    # Random Sample
    random_outcome = np.random.choice(outcomes, p=outcome_probs)

    return random_outcome


# Fair Coin Flip or Multiple Coins
def flip_fair_coin(num_coins=None):
    """
    Returns a fair coin flip.
    :return: Either string 'Head' or 'Tails'
    """
    finite_prob_space = {'heads': 0.5, 'tails': 0.5}
    if num_coins:
        return [sample_from_finite_probability_space(finite_prob_space) for _ in range(num_coins)]
    return sample_from_finite_probability_space(finite_prob_space)


# Plot Discrete histogram
def discrete_histogram(array, freq=False, figsize=(5, 4)):
    """
    Plots a discrete histogram given a 1D array of values
    :param array: 1D array consisting of data
    :param freq: boolean (True => plot frequencies, False => plot counts)
    :param figsize: tuple (width, height) of how large to make the plotted figure
    """
    array_as_series = pd.Series(array)
    counts = array_as_series.value_counts().sort_index()

    if freq:
        counts /= counts.sum()

    plt.figure(figsize=figsize)
    plt.xlabel('Value')
    if freq:
        plt.ylabel('Frequency')
    else:
        plt.ylabel('Count')

    axis = counts.plot(kind='bar')
    figure = axis.get_figure()
    figure.autofmt_xdate()  # Rotates x-axis labels to be more readable

    plt.tight_layout()  # Tidy up and remove some margins


# Probability Table (1D Array)
def print_prob_table_array(probs, outcomes):
    """
    Prints a probability table that is stored as a 1D array
    :param probs: a 1D array of non-negative entries that add to 1
    :param outcomes: List of labels; i-th label for i-th entry
    :return: Nothing
    """
    if len(probs) != len(outcomes):
        raise Exception('The number of outcomes and number of probabilities must match')
    print(pd.Series(probs, outcomes))


# Join Probability Table
def print_joint_prob_table_dict(dict_in_dict):
    """
    Prints a join probability table that is stored using the dictionaries within a dictionary representation
    :param dict_in_dict: Joint probability table stored as dictionaries within a dictionary
    :return: Nothing
    """
    print(pd.DataFrame(dict_in_dict).T)


# Joint Probability Table Array
def print_joint_prob_table_array(array, row_labels, col_labels):
    """
    Prints a joint probability table that is stored using 2D array representation
    :param array: 2D Array for join probability table
    :param row_labels: list of labels, i-th label is for the i-th row in <array>
    :param col_labels: list of labels, i-th label is for the i-th column in <array>
    :return: Nothing
    """
    if len(array.shape) != 2:
        raise Exception('The array specified must be 2D')
    print(pd.DataFrame(array, row_labels, col_labels))


# Testing Ground
# 100 flips of fair coin
flips = flip_fair_coin(num_coins=100)

# Histogram
discrete_histogram(flips)
plt.show()

# Histogram with Frequency Y-axis
discrete_histogram(flips, freq=True)
plt.show()

# Plotting Fraction of heads as function of number of flips (100k flips)
n = 100_000
heads_so_far = 0
fraction_of_heads = []
for i in range(n):
    if flip_fair_coin() == 'heads':
        heads_so_far += 1
    fraction_of_heads.append(heads_so_far/(i+1))    # Fraction says fraction of heads after the first "i" tosses

plt.figure(figsize=(8, 4))
plt.plot(range(1, n+1), fraction_of_heads)
plt.xlabel('Number of flips')
plt.ylabel('Fraction of Heads')
plt.show()

# Representing Joint Probability Table
# Method 1, Basic
prob_table = {('sunny', 'hot'): 3/10,
              ('sunny', 'cold'): 1/5,
              ('rainy', 'hot'): 1/30,
              ('rainy', 'cold'): 2/15,
              ('snowy', 'hot'): 0,
              ('snowy', 'cold'): 1/3}

# Method 2, Dict in Dict
prob_W_T_dict = {}
for w in {'sunny', 'rainy', 'snowy'}:
    prob_W_T_dict[w] = {}
prob_W_T_dict['sunny']['hot'] = 3/10
prob_W_T_dict['sunny']['cold'] = 1/5
prob_W_T_dict['rainy']['hot'] = 1/30
prob_W_T_dict['rainy']['cold'] = 2/15
prob_W_T_dict['snowy']['hot'] = 0
prob_W_T_dict['snowy']['cold'] = 1/3
print_joint_prob_table_dict(prob_W_T_dict)

# Method 3, 2D Arrays & DataFrames
prob_W_T_rows = ['sunny', 'rainy', 'snowy']
prob_W_T_cols = ['hot', 'cold']
prob_W_T_array = np.array([[3/10, 1/5], [1/30, 2/15], [0, 1/3]])
print_joint_prob_table_array(prob_W_T_array, prob_W_T_rows, prob_W_T_cols)
