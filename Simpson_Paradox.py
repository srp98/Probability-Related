# Simpson's Paradox Problem
# Import Required Lib's
import numpy as np

prob_space = {
    ('female', 'A', 'admitted'): 0.019566946531153304,
    ('female', 'A', 'rejected'): 0.004295183384887301,
    ('female', 'B', 'admitted'): 0.0037560760053027007,
    ('female', 'B', 'rejected'): 0.0017675651789660005,
    ('female', 'C', 'admitted'): 0.044547061422890007,
    ('female', 'C', 'rejected'): 0.086473707467962915,
    ('female', 'D', 'admitted'): 0.028999116217410508,
    ('female', 'D', 'rejected'): 0.053855501546619514,
    ('female', 'E', 'admitted'): 0.020839593460008802,
    ('female', 'E', 'rejected'): 0.065992045956694709,
    ('female', 'F', 'admitted'): 0.0052739726027397011,
    ('female', 'F', 'rejected'): 0.070068493150684918,
    ('male', 'A', 'admitted'): 0.11301369863013702,
    ('male', 'A', 'rejected'): 0.069266460450729109,
    ('male', 'B', 'admitted'): 0.077949624392399511,
    ('male', 'B', 'rejected'): 0.045779938135218703,
    ('male', 'C', 'admitted'): 0.026568714096332307,
    ('male', 'C', 'rejected'): 0.045238621299160404,
    ('male', 'D', 'admitted'): 0.030404330534688506,
    ('male', 'D', 'rejected'): 0.061730004418912916,
    ('male', 'E', 'admitted'): 0.011816173221387503,
    ('male', 'E', 'rejected'): 0.030384445426425107,
    ('male', 'F', 'admitted'): 0.0049447635881573011,
    ('male', 'F', 'rejected'): 0.077467962881131211
}

# Labels
gender_labels = ['female', 'male']  # axis 0
department_labels = ['A', 'B', 'C', 'D', 'E', 'F']  # axis 1
admission_labels = ['admitted', 'rejected']  # axis 2

# Mapping Labels to 0 and 1
gender_mapping = {label: index for index, label in enumerate(gender_labels)}
department_mapping = {label: index for index, label in enumerate(department_labels)}
admission_mapping = {label: index for index, label in enumerate(admission_labels)}

# Join Prob. Table
joint_prob_table = np.zeros((2, 6, 2))

# Full prob. space into joint prob. table
for gender, department, admission in prob_space:
    joint_prob_table[gender_mapping[gender],
                     department_mapping[department],
                     admission_mapping[admission]] = prob_space[(gender,
                                                                 department,
                                                                 admission)]

# Paradox, Let's look at prob. women admitted vs prob. men admitted
# Since numpy 3D array, axis(0) is gender, axis(1) is department and axis(2) is admission decision
# So, Marginalization of department axis(1), gives joint prob. table of r.v's gender and admission
joint_prob_gender_admission = joint_prob_table.sum(axis=1)

# Probability of woman applies and is admitted by university
print(joint_prob_gender_admission[gender_mapping['female'], admission_mapping['admitted']])

# Conditioning to see prob. of being admitted given that applicant is female
# Let's restrict the join prob table of G and A so we only look at when G = female
female_only = joint_prob_gender_admission[gender_mapping['female']]
# Normalize Vector to sum it to 1
prob_admission_given_female = female_only/np.sum(female_only)
# Convert result to dict format
prob_admission_given_female_dict = dict(zip(admission_labels, prob_admission_given_female))
print(prob_admission_given_female_dict)


# Probability of woman applies and is admitted by university
print(joint_prob_gender_admission[gender_mapping['male'], admission_mapping['admitted']])

# Conditioning to see prob. of being admitted given that applicant is male
# Let's restrict the join prob table of G and A so we only look at when G = male
male_only = joint_prob_gender_admission[gender_mapping['male']]
# Normalize Vector to sum it to 1
prob_admission_given_male = male_only/np.sum(male_only)
# Convert result to dict format
prob_admission_given_male_dict = dict(zip(admission_labels, prob_admission_given_male))
print(prob_admission_given_male_dict)

# There do seem to be a bias b/w female and male admissions, We can investigate how admissions are in each department
# Since we already have join prob of G and A, lets condition based on being admitted
admitted_only = joint_prob_gender_admission[:, admission_mapping['admitted']]

# Conditional Prob. of gender given admitted
prob_gender_given_admitted = admitted_only/np.sum(admitted_only)
prob_gender_given_admitted_dict = dict(zip(gender_labels, prob_gender_given_admitted))
print(prob_gender_given_admitted_dict)

# Let's Condition on both G and D taking on specific values together to determine the prob. of being admitted
# So, prob of being admitted given gender and department
for department in department_labels:
    for gender in gender_labels:
        restricted = joint_prob_table[gender_mapping[gender], department_mapping[department]]
        print(department, gender, dict(zip(admission_labels, restricted / np.sum(restricted)))['admitted'])
