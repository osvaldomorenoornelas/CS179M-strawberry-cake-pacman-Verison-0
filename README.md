# Team strawberry-cake PacMan-ctf

## Team Members - Paris Hom and Osvaldo Moreno Ornelas

## Python3 version of UC Berkeley's CS 188 PacMan Capture the Flag project

### Original Licensing Agreement (which also extends to this version)

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The PacMan AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

### This version attribution

This version (cshelton/PacMan-ctf github repo) was modified by Christian
Shelton (cshelton@cs.ucr.edu) on June 23, 2020 to run under Python 3.

## Important Note

**Main Branch:** The code has been merged and tested in the branch under the name "V1-Osvaldo_Dev". Please refer to this branch to see our work and test our code.

## Running the Code

**Required Packages:** tensorflow, torch, numpy

**Necessary Files:** DQAgent.py, DQNetwork.py, myTeam.py, trainingOutput.pickle, and trainingInput.pickle

- Make sure you have the required packages and necessary files
- run "python capture.py -r DQAgent" to run our agent as the red team
- run "python capture.py -b DQAgent" to run our agent as the blue team

## Agents Strategy

### Defensive Agent

The Defensive agent implementation is in myTeam.py. The class used is DefensiveReflexAgent.

If there are no enemies on the team's side, the Defensive Agent will move toward the middle of the map and wait until an enemy has entered our side. If the Defensive Agent cannot see the enemy, then it will move randomly until it sees the enemy or until the enemy eats one of the team’s food pellets. If the enemy eats a food pellet, then the Defensive Agent will move to the coordinates of our team’s most recently eaten food pellet. This is because it is likely that the enemy will be around the area of the most recently eaten food pellet. Food pellets are usually placed near each other, which means that the enemy is also likely to stay at that location to eat the rest of the food pellets in that location. Thus going to the most recently eaten food pellet will help the Defensive Agent locate the enemy.

Additionally, we store the most recent position of the enemy PacMan agent if it is available. This helps our Defensive Agent locate the enemy after the Defensive Agent dies. We did this because it is likely that the enemy is still near the position it was last at.

If the Defensive agent is scared, it will try to get eaten by the enemy PacMan if the distance to from the start position to current position is greater than the number of remaining scared moves. Otherwise, the Defensive agent will avoid getting eaten by the enemy PacMan.

### Offensive Agent

The Offensive agent implementation is in DQAgent.py. The class used is DQAgent.
In this class, the Offensive agent uses a Neural Network to compute the best action to take, given the current state of the game.
The Offensive Agent's Neural Network takes in four features and outputs the QValue of those four features. This is what the Offensive Agent will use to calculate the best action to take given the current game state.

The four features that the Offensive Agent uses are: distance to enemy, distance to food, distance to our team's side, and the current score of the game.

The decision to use these four features are as follows:

- distance to enemy: the Offensive Agent should know how far or close the enemies are. This will help the Offensive Agent try to avoid going to places where the enemy is near and try to survive longer. This will also prevent the Offensive agent from wasting time going back to the opponent's side of the board.
- distance to food: the main goal of the Offensive Agent is to increase the team's points to win the game. In order to do that, the Offensive Agent needs to know where the food is.
- distance to our team's side: in order to increase our team's points, the Offensive Agent has to actually bring the food back to our team's side. This feature will help the Agent figure out what actions will help it get back home (by reducing the distance to our team's side)
- current score of the game: this feature tells the Offensive Agent whether the team is winning, losing, or at a tie. We added this feature because we thought that it would help the Offensive Agent decide whether it should go back home more food or less food. For example, if the game is currently at a tie, then it suffices to bring back one food pellet to start winning the game.

### Decision Making

- To make a decision, the agent creates a DQNetwork object with the desired inputs, outputs, and layers to create the model
- Upon creating the object, the data is loaded into tensors
- The data is passed into the train model module
- The data is trained and the agents are launched
- The agent gathers the features and values
- The features are used in the function chooseAction where the Offensive Agent makes a decision of where to move
- In order to make the best decision, the Offensive Agent uses the function bestActionNN
- The function gets the possible actions and runs it in a sequential evaluation
- The features for that action are passed into the model by using the function getNetworkPrediction, who calls the DQNetwork function predict
- The function predicts the value based on the features passed into the model and returns the result
- In the end, the action with the highest value gets chosen


The Neural Network's training data is gathered from Agent0 Offensive Agent. This is because Agent0 worked well against the baseline team. This also helped speed up the process of getting our Offensive Agent to learn because it allowed us to avoid having our agent begin learning by performing random actions. The Offensive Agent could instead learn off our Agent0 agent that already worked well.

Additionally, the Offensive Agent has a policy implemented to return home after carrying 20% of the food needed to eat to win the game. This is to ensure that our team will always win by retrieving slightly more food than the enemy.

## Decision to pursue this strategy

We decided to use this strategy because we wanted to make the Defensive Agent a good defender of the food while having the Offensive agent retrieve food to score the team points. We also wanted the Defensive Agent to stay in the middle of the map while there are no enemies on our team's side because we realized that all enemies would have to cross the middle to enter our team's side. By having our Agent in the middle, this will help locate and eat the enemies quickly.

A Neural Network was decided on for the Offensive Agent because in previous Milestones, the Offensive Agent would never try to bring food back to our team's side. This would result in the team getting a tie or loss. We thought that this was because the decision to bring food back may have been too complex for the agent to decide on. We thought that a Neural Network would help solve this problem using more hidden layers.

## How the work was divided

Paris and Osvaldo worked on the Offensive agent together. Osvaldo implemented the Neural Network using PyTorch and Paris got the training data from Agent0 and found features for the Neural Network.

## How well the agent works

The Agents were both tested on four maps: defaultCapture, jumboCapture, fastCapture, and bloxCapture. This is because the four maps are very different in terms of size and number of pellets. We thought that if our Agents are able to perform well in these four maps, then it means that our agents are able to generalize its environment and make moves that are still optimal.

- defaultCapture: default map.
- jumboCapture: large and a lot of pellets
- fastCapture: small and not many pellets
- bloxCapture: medium in size and number of pellets

**Defensive Agent:** 
The way the Defensive Agent was evaluated was by setting the Offensive Agent to not move and stay at the starting position. This was done to see how well the Defensive Agent is able to defend the food and eat the enemy. Our Defensive Agent never tries to retrieve food and score points for the team, so our team would only see losses or ties. If our team loses, this means that the Defensive Agent does not do a good job at defending food. If our team tied, then this means that the Defensive Agent was able to defend food well enough to prevent to enemy team from scoring.
This was tested across 80 games against the Baseline team: 10 games each as red and blue team  on the four maps listed above. On the four maps, there was a 100% tie rate. This means that the Defensive Agent does very well defending the team's food against the BaseLine team.
|                | DQAgent VS Baseline, 10 games each map   |               |             |
|----------------|:----------------------------------------:|---------------|-------------|
| Map            | DQAgent Team                             | Average Score | Win Percent |
| defaultCapture | red                                      | 1.8           | 90%         |
| defaultCapture | blue                                     | -1.8          | 90%         |
| bloxCapture    | red                                      | 8.6           | 90%         |
| bloxCapture    | blue                                     | -18           | 90%         |
| jumboCapture   | red                                      | 17.4          | 90%         |
| jumboCapture   | blue                                     | -15.6         | 100%        |
| fastCapture    | red                                      | 0.9           | 80%         |
| fastCapture    | blue                                     | -1            | 80%         |

**Offensive Agent:**  This was tested across 80 games against the Baseline team: 10 games on the four map listed above each as red and blue team. We also used the same number of data points (30k data points) for every game. This would ensure that the Offensive Agent has the same training data and makes similar decisions.
On the four maps, there was at least an 80% win rate.

|                | DQAgent VS Baseline, 10 games each map   |               |             |
|----------------|:----------------------------------------:|---------------|-------------|
| Map            | DQAgent Team                             | Average Score | Tie Percent |
| defaultCapture | red                                      | 0             | 100%        |
| defaultCapture | blue                                     | 0             | 100%        |
| bloxCapture    | red                                      | 0             | 100%        |
| bloxCapture    | blue                                     | 0             | 100%        |
| jumboCapture   | red                                      | 0             | 100%        |
| jumboCapture   | blue                                     | 0             | 100%        |
| fastCapture    | red                                      | 0             | 100%        |
| fastCapture    | blue                                     | 0             | 100%        |

## Lessons learned from the iteration

We learned how to implement a Neural Network and feature selection. Before this class, we never implemented a Neural Network before. So, to have the Offensive Agent run with the Neural Network and win matches is something that we are proud of. We learned how to implement a Neural Network using PyTorch.

In the beginning, our Neural Network took in 8 features. However, these features did not work well for the Offensive Agent and the Offensive Agent performed poorly. We learned to figure out what features are important and how many features we should use.