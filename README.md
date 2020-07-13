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

### Important Note

**Main Branch:** The code has been merged and tested in the branch under the name "Agent0". Please refer to this branch to see our work and test our code.

**Workflow:** The main work was done in a different fork "tomehomme/CS179M-strawberry-cake-pacman-ctf". The version0 fork "osvaldomorenoornelas/CS179M-strawberry-cake-pacman-Verison-0" was created to turn in the work. Please refer to the other repository in order to see more commits and work stats.


## Agents Stragety
The agents both have an aggressive strategy.
#### Ghost Agent
For the Ghost agent, the class used is DefensiveReflexAgent. 

In this class, we store the most recent position of the enemy PacMan agent if it is available. Otherwise, the DefensiveReflexAgent will wander until it find the enemy agent or when food on our side has been eaten. If our food has been eaten, our Ghost agent will head toward the location of the most recently eaten food.

If the Ghost agent is scared, it will try to get eaten by the enemy PacMan if it has been scared for less than 20 moves. Otherwise, the Ghost agent will avoid getting eaten by the enemy PacMan.
#### PacMan Agent
The PacMan agent uses OffensiveReflexAgent. 

In this class, we always try to seek the closest food. When eating we keep a counter. When the counter reaches 4, we go back and dump the food back to base. If the PacMan is killed while carrying 5 orbs, the PacMan goes back to the place the orbs were dumped. 

Also, the PacMan remembers the decisions that have been made and when deciding where to go, it tries to go to a better direction then it did last time, rather than making a random decision.

### Decision to pursue this strategy
We decided to use this strategy because we wanted to make the Ghost a good defender of the food while having the PacMan agent launch an offensive that took small amounts of risk at a time.


### How the work was divided
Paris worked on the Ghost agent while Osvaldo worked on the PacMan agent. Paris helped Osvaldo in the development of the PacMan agent.

#### How well the agent works
The Agents were tested by running against the Baseline Team 100 times: 50 times as Red team and 50 times as Blue team. On these 100 trials, there was a 90% win rate, 2% tie, and 8% loss as the Red team and 100% win rate as the Blue team. On Average, the team wins by an average of 8.62 points as the Red team, and wins by an average of 11.78 points as the Blue team.

**Ghost:** Ghost is capable of quickly locating the enemy PacMan and pursuing it using the path that results in the shortest maze distance. As soon as the enemy eats a food pellet, the Ghost can try to chase after the PacMan. Some work that can be done is having the Ghost be able to locate the enemy PacMan agent as soon as they enter our side.

**PacMan:**  PacMan retrieves the food as expected. PacMan makes decisions based on a bank. The decisions are not as smart or dynamic as expected, there is still work to do in that aspect. PacMan "dances" around Ghost if there is one always trying to block. It is a work in progress to make PacMan flank the Ghost.

#### Lessons learned from the iteration

**To Change:** PacMan's decision making algorithm into something more dynamic and self-learning. The Ghost does not need to change much. But, the Ghost agent can improve on locating where the enemy PacMan is as soon as they enter our side, instead of waiting until the opponent PacMan consumes our food.

**To Keep:** One thing that can be kept for the PacMan (offensive) agent is the way the food is retrieved and brought back. The food dumped to go back and get it back if it is higher than a set limit.
As for the Ghost (defensive) agent, we will keep how the Ghost goes to the location of the most recently eaten food when it does not have the coordinates of the enemy PacMan. We will also keep how the Ghost saves the most recent location of the enemy PacMan agent.