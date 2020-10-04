#!/bin/python3
import os
import json
import numpy as np 
import cvxpy as cp 
from itertools import chain

if not os.path.exists("outputs"):
    os.makedirs("outputs")

def save_json(to_write):
    file = open("./outputs/output.json", "w")
    json.dump(to_write, file)
    file.close()

# Initialization
arr = [1/2, 1, 2]
team_number = 5
y = arr[team_number % 3]
step_cost = -10/y


# Information about states
no_states = 60
alpha = np.zeros(no_states)
alpha[no_states-1] = 1 # Probability of being in state 5,4,3 initially is 1
alpha = alpha.transpose()
A = []
R = []
X = []
states = []
for md_health in range(5):
    for no_arrows in range(4):
        for hero_health in range(3):
            states.append([md_health, no_arrows, hero_health])


tuples = np.zeros(no_states)
actions = ["SHOOT", "DODGE", "RECHARGE", "NOOP"]
possible_actions = []

# Formulating A matrix
# Actions in the order SHOOT, DODGE, RECHARGE, NOOP
def generate_A_and_R():
    for md_health in range(5):
        for no_arrows in range(4):
            for hero_health in range(3):
                
                if md_health and no_arrows and hero_health: # SHOOT
                    R.append(step_cost) # Reward for this action
                    vec = np.zeros(no_states)  
                    vec[states.index([md_health, no_arrows, hero_health])] = 1   
                    vec[states.index([md_health-1, no_arrows-1, hero_health-1])] = -0.5
                    vec[states.index([md_health, no_arrows-1, hero_health-1])] = -0.5
                    A.append(vec.tolist())
                    tuples[states.index([md_health, no_arrows, hero_health])] += 1
                    possible_actions.append(actions[0])                   
                    

                if md_health and hero_health: # DODGE
                    R.append(step_cost) # Reward for this action
                    vec = np.zeros(no_states)
                    vec[states.index([md_health, no_arrows, hero_health])] = 1 

                    if hero_health == 2:
                        if no_arrows == 3:
                            vec[states.index([md_health, no_arrows, hero_health-1])] = -0.8
                            vec[states.index([md_health, no_arrows, hero_health-2])] = -0.2
                            
                        else:
                            vec[states.index([md_health, no_arrows + 1, hero_health-1])] = -0.64
                            vec[states.index([md_health, no_arrows + 1, hero_health-2])] = -0.16
                            vec[states.index([md_health, no_arrows, hero_health-1])] = -0.16
                            vec[states.index([md_health, no_arrows, hero_health-2])] = -0.04
                                                       
                    elif hero_health == 1:
                        if no_arrows == 3:
                            vec[states.index([md_health, no_arrows, hero_health-1])] = -1
                                                       

                        else:
                            vec[states.index([md_health, no_arrows+1, hero_health-1])] = -0.8
                            vec[states.index([md_health, no_arrows, hero_health-1])] = -0.2
                                                       
                    tuples[states.index([md_health, no_arrows, hero_health])] += 1
                    possible_actions.append(actions[1])                   
                    A.append(vec.tolist())

                if md_health and hero_health != 2: # RECHARGE
                    R.append(step_cost) # Reward for this action
                    vec = np.zeros(no_states)
                    vec[states.index([md_health, no_arrows, hero_health])] = 1
                    vec[states.index([md_health, no_arrows, hero_health+1])] = -0.8 
                    vec[states.index([md_health, no_arrows, hero_health])] += -0.2
                    tuples[states.index([md_health, no_arrows, hero_health])] += 1  
                    possible_actions.append(actions[2])                   
                    A.append(vec.tolist())

                if md_health == 0:
                    R.append(0) # Reward for this action
                    vec = np.zeros(no_states)
                    vec[states.index([md_health, no_arrows, hero_health])] = 1
                    tuples[states.index([md_health, no_arrows, hero_health])] += 1
                    possible_actions.append(actions[3])                   
                    A.append(vec.tolist())


def generate_policy():
    start = 0
    end = 0
    policy = []
    for md_health in range(5):
        for no_arrows in range(4):
            for hero_health in range(3):
                index = np.argmax(X[int(start):start + int(tuples[end])])
                policy.append([(md_health, no_arrows, hero_health), possible_actions[start + index]])
                start += int(tuples[end])
                end += 1
                
    return policy
                
def action_total():
    sum = 0
    for item in tuples:
        sum += item
    return int(sum)
                
def lpp():
    global X
    size = action_total()
    X = cp.Variable(shape=(size, 1), name = "X")
    constraints = [cp.matmul(A, X) == alpha, X >= 0]
    objective = cp.Maximize(cp.matmul(R,X))
    problem = cp.Problem(objective, constraints)
    return problem.solve()

generate_A_and_R()

# Make them all correct numpys to send to LPP solver

A = np.array(A)
A = A.transpose()
R = np.array(R)
R = R[:, np.newaxis]
R = R.transpose()
alpha = np.array(alpha)
alpha = alpha[:, np.newaxis]

# Solve the LPP
objective = lpp()

# Format X
X = X.value
X = list(chain.from_iterable(X))


# Generate best policy using X, tuples and possible_actions
policy = generate_policy()

to_write = {
    "a": A.tolist(),
    "r": R.tolist(),
    "alpha": alpha.tolist(), 
    "x": X,
    "policy": policy,
    "objective": objective
}

save_json(to_write)
