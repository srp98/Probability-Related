# Robot Localization Example
The progrom contains a simple graphical robot localization example using Hidden Markov Model

## Requirements
- Python 3.8+
- Numpy
- Scipy
- Matplotlib
- tkinter

## Dataset
The folder `Data` contains the required dataset for this test

## Problem Statement
### Model 
The robot position at time $i$ is given by a random variable $Z_i$ which takes on values $\{0, 1,\dots, 11\} \times \{0, 1,\dots, 7\}$, example $Z_2 = (5, 4)$ means at timestep 2, the robot is in column 5 and row 4. It's actions (up, down, left, right, stay) depend on 3 factors as in HMMM's prior distribution, observation state, trasition state (previous state)

We need to treat the boundary of the grid differently. For instance, at the top of the grid, the robot can't go any higher. We'll renormalize the probabilities of its remaining actions at the top so that they sum to 1. Such boundary cases suggest that the transition probabilities depend on the robot's current location and its previous action. Thus, we model the robot's hidden state $X_ i$ at time $i$ to consist of both its location $Z_ i$ and the action $A_ i$ it last took to reach $Z_ i$. 

We have access to a noisy sensor that puts a uniform distribution on valid grid positions within 1 grid cell of the robot's current true position. Also, this noisy sensor only gives a guess as to the robot's current position and tells us nothing about what actions our robot has taken. In other words, at time i, we observe $Y_ i$, which takes on a value in $\{ 0,1,\dots ,11\} \times \{ 0,1,\dots ,7\}$

Lastly, we shall assume that initially the robot is in any of the grid locations with equal probability and always starts with its action set to ‘stay'. Program `robot.py` generates the initial distribution, for every timestep $i$ so we implement `forward-backword` algorithm to compute marginal distribution $p_{X_ i|Y_0,\dots ,Y_{n-1}}(\cdot |y_0,\dots ,y_{n-1})$ for each $i$ for the function in `inference.py`

The forward-backward algorithm takes takes $\mathcal{O}(nk^2)$ computations, where $n$ is the total number of time steps, and $k$ is the number of possible hidden states. However, there is actually additional structure such that it's possible to compute all the node marginals in roughly $\mathcal{O}(nk)$ computations— $\mathcal{O}(nk)$ for pre-processing and $\mathcal{O}(n)$ computations for the actual message passing as shown in `inference.py`.
We also have to handle missing observations in this algorithm

We implement `Viterbi` algorithm aswell to know which *sequence* of states the robot was most likely to have visited using MAP estimate of the entire sequence of hidden states given a sequence of observations. This algorithm also needs to handle missing observations if encountered. We can also modify the program to out put the second-best solution to the MAP problem rather than best solution as shown in the function `second_best` in `inference.py`

Note: If the MAP estimate is not unique (i.e., there are multiple sequences that all achieve the highest maximum posterior probability), then a second-best sequence should be one of the other equally good MAP estimates.

### Running Program
`graphics.py` and `robot.py` are programs for generating graphics and initial generation of probabilistic distributions for the robot's position and to the run the program just run `inference.py` to view result and output.
