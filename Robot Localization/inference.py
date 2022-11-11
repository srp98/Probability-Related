#!/usr/bin/env python
# inference.py
import collections
import sys
import graphics
import numpy as np
import robot

# Variables from robot.py for the Robot Model
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# Function helpful for computing logs without yielding a NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# Forward Backward Algorithm
def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution should be encoded as a Distribution (see the
    Distribution class in robot.py and see how it is used in both robot.py and the function generate_data(), and the
    i-th Distribution should correspond to time step i
    """
    num_time_steps = len(observations)
    # forward_messages[i] is P([state at i] and y's before i)
    forward_messages = [prior_distribution]

    for t in range(num_time_steps - 1):
        # Incorporate t-th observation
        if observations[t] is not None:
            prob_given_y = robot.Distribution()     # Our Modified Dictionary
            for state, prob in forward_messages[t].items():
                prob_given_y[state] = prob * observation_model(state)[observations[t]]
            prob_given_y.renormalize()

        else:
            prob_given_y = forward_messages[t]

    # Take a step forward in time, using the transition model.
    next_dist = robot.Distribution()
    for state, prob in prob_given_y.items():
        for new_state, trans_prob in transition_model(state).items():
            next_dist[new_state] += prob * trans_prob
    next_dist.renormalize()
    forward_messages.append(next_dist)

    backward_messages = [None] * num_time_steps
    # Reverse transition dictionary
    backwards_transitions = collections.defaultdict(robot.Distribution)
    for state_1 in all_possible_hidden_states:
        for state_2 in all_possible_observed_states:
            backwards_transitions[state_2][state_1] += prob

    # backward messages[i] is P(y's after i | state at i)
    backward_messages = [None] * num_time_steps
    uniform = robot.Distribution()
    for s in all_possible_hidden_states:
        uniform[s] = 1
    uniform.renormalize()
    backward_messages[-1] = uniform
    for t in range(num_time_steps - 1, 0, -1):
        if observations[t] is not None:
            prob_given_y = robot.Distribution()
            for state, prob in backward_messages[t].items():
                prob_given_y[state] = prob * observation_model(state)[observations[t]]
            prob_given_y.renormalize()
        else:
            prob_given_y = backward_messages[t]

        prev_dist = robot.Distribution()
        for state, prob in prob_given_y.items():
            for past_state, trans_prob in backwards_transitions[state].items():
                prev_dist[past_state] += prob * trans_prob
        prev_dist.renormalize()
        backward_messages[t - 1] = prev_dist

    marginals = []
    for t in range(num_time_steps):
        marginal_t = robot.Distribution()
        for state in all_possible_hidden_states:
            product = forward_messages[t][state] * backward_messages[t][state]
            if observations[t] is not None:
                product *= observation_model(state)[observations[t]]
            if product > 0:
                marginal_t[state] = product
        marginal_t.renormalize()
        marginals.append(marginal_t)

    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state (a missing observation is encoded as None)

    Output
    ------
    A list of estimated hidden states, each encoded as a tuple (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)

    # forward[t][s] represents the MAP estimate of state of the robot in timestep t, assuming robot is in state s.
    # It contains (log prob, old_s) the latter of which points to state in time t-1 that is part of MAP estimate
    forward = []
    initial_map = {}
    for s in all_possible_hidden_states:
        initial_map[s] = [careful_log(prior_distribution[s]), None]
        if observations[0] is not None:
            initial_map[s][0] += careful_log(observation_model(s)[observations[0]])
    forward.append(initial_map)

    for t in range(1, num_time_steps):
        # In each timestep, first apply transition probabilities from the previous timestep.
        new_map = collections.defaultdict(lambda: (-np.inf, None))
        for old_s, (old_prob, _) in forward[-1].items():
            for new_s, trans_prob in transition_model(old_s).items():
                total_prob = old_prob + careful_log(trans_prob)
                # Update new MAP estimate if necessary.
                if total_prob > new_map[new_s][0]:
                    new_map[new_s] = (total_prob, old_s)

        # Then, apply observation probabilities.
        if observations[t] is not None:
            for s in all_possible_hidden_states:
                log_prob, back_ptr = new_map[s]
                new_map[s] = (log_prob +
                              careful_log(observation_model(s)[observations[t]]),
                              back_ptr)
        forward.append(new_map)

    # Read off the answer using back pointers
    MAP_Estimate = [None] * num_time_steps
    MAP_Estimate[num_time_steps - 1] = max(forward[num_time_steps - 1], key=lambda i: forward[num_time_steps - 1][i][0])
    for t in range(num_time_steps - 2, -1, -1):
        MAP_Estimate[t] = forward[t + 1][MAP_Estimate[t + 1]][1]

    return MAP_Estimate


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state (a missing observation is encoded as None)

    Output
    ------
    A list of estimated hidden states, each encoded as a tuple (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)

    # Basically for each (possible) hidden state at time step i, we need to keep track of the best previous hidden state
    # AND the second best previous hidden state--where we need to keep track of TWO back pointers per (possible) hidden
    # state at each time step!
    messages = []   # Best values so far
    messages2 = []  # Second Best values so far
    back_pointers = []  # Per timestep per hidden state, we now need 2 back pointers

    # Fold Observations into singleton potentials
    phis = []   # phis[n] is singleton potential for node n
    for n in range(num_time_steps):
        potential = robot.Distribution()
        observed_state = observations[n]
        if n == 0:
            for hidden_state in prior_distribution:
                value = prior_distribution[hidden_state]
                if observed_state is not None:
                    value *= observation_model(hidden_state)[observed_state]
                if value > 0:  # only store entries with nonzero prob.
                    potential[hidden_state] = value
        else:
            for hidden_state in all_possible_hidden_states:
                if observed_state is None:
                    # singleton potential should be identically 1
                    potential[hidden_state] = 1.
                else:
                    value = observation_model(hidden_state)[observed_state]
                    if value > 0:  # only store entries with nonzero prob.
                        potential[hidden_state] = value
        phis.append(potential)

    # Forward Pass
    # Handle each timestep differently
    initial_message = {}
    for hidden_state in prior_distribution:
        value = -careful_log(phis[0][hidden_state])
        if value < np.inf:  # only store entries with nonzero prob.
            initial_message[hidden_state] = value
    messages.append(initial_message)
    initial_message2 = {}  # there is no second-best option
    messages2.append(initial_message2)

    # rest of the time steps
    for n in range(1, num_time_steps):
        prev_message = messages[-1]
        prev_message2 = messages2[-1]
        new_message = {}
        new_message2 = {}
        new_back_pointers = {}  # need to store 2 per possible hidden state

        # only look at possible hidden states given observation
        for hidden_state in phis[n]:
            values = []
            # each entry in values will be a tuple of the form: (<value>, <previous hidden state>,
            # <which back pointer we followed>), where <which back pointer we followed> is 0 (best back pointer)
            # or 1 (second-best back pointer)

            # iterate through best previous values
            for prev_hidden_state in prev_message:
                value = prev_message[prev_hidden_state] - \
                        careful_log(transition_model(prev_hidden_state)[hidden_state]) - \
                        careful_log(phis[n][hidden_state])
                if value < np.inf:
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 0))

            # also iterate through second-best previous values
            for prev_hidden_state in prev_message2:
                value = prev_message2[prev_hidden_state] - \
                        careful_log(transition_model(prev_hidden_state)[hidden_state]) - \
                        careful_log(phis[n][hidden_state])
                if value < np.inf:
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 1))

            if len(values) > 0:
                # this part could actually be sped up by not using a sorting algorithm...
                sorted_values = sorted(values, key=lambda x: x[0])
                best_value, best_prev_hidden_state, which_back_pointer = sorted_values[0]

                # for the best value, the back pointer should *always* be 0, meaning that we follow the best back
                # pointer and not the second best
                if len(values) > 1:
                    best_value2, best_prev_hidden_state2, which_back_pointer2 = sorted_values[1]
                else:
                    best_value2 = np.inf
                    best_prev_hidden_state2 = None
                    which_back_pointer2 = None

                new_message[hidden_state] = best_value
                new_message2[hidden_state] = best_value2
                new_back_pointers[hidden_state] = \
                    ((best_prev_hidden_state, which_back_pointer),
                     (best_prev_hidden_state2, which_back_pointer2))

        messages.append(new_message)
        messages2.append(new_message2)
        back_pointers.append(new_back_pointers)

    # Backward Pass
    # handle last time step differently
    values = []
    for hidden_state, value in messages[-1].items():
        values.append((value, hidden_state, 0))
    for hidden_state, value in messages2[-1].items():
        values.append((value, hidden_state, 1))

    divergence_time_step = -1

    if len(values) > 1:
        # this part could actually be sped up by not using a sorting algorithm...
        sorted_values = sorted(values, key=lambda x: x[0])

        second_best_value, hidden_state, which_back_pointer = sorted_values[1]
        estimated_hidden_states = [hidden_state]

        # rest of the time steps
        for t in range(num_time_steps - 2, -1, -1):
            hidden_state, which_back_pointer = back_pointers[t][hidden_state][which_back_pointer]
            estimated_hidden_states.insert(0, hidden_state)
    else:
        # this happens if there isn't a second best option, which should mean that the only possible option
        # (the MAP estimate) is the only solution with 0 error
        estimated_hidden_states = [None] * num_time_steps

    return estimated_hidden_states


# Generating data from the hidden Markov model
def generate_data(num_time_steps, make_some_observations_missing=False, random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if random seed is not None, then this makes the randomness deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# Main
def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = generate_data(num_time_steps,
                                                    make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestamp = 2
    print("Most likely parts of marginal at time %d:" % timestamp)
    if marginals[timestamp] is not None:
        print(sorted(marginals[timestamp].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " + "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " + ", ".join(["%d" % time_step for time_step in
                                                                           difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " + "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " + ", ".join(["%d" % time_step for time_step in
                                                                           difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " + "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " + ", ".join(["%d" % time_step for time_step in
                                                                           difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
