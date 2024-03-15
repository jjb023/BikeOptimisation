# BikeOptimisation

## Similar problem papers
- https://www.sciencedirect.com/science/article/pii/S0303264716300491 Evaluating clustering methods within the Artificial Ecosystem Algorithm and their application to bike redistribution in London
- https://link.springer.com/article/10.1007/s12293-012-0101-3 Automated self-organising vehicles for Barclays Cycle Hire

## Potential visuals
- Average trips at each time of day with mean and standard deviation. (Normal vs tube station day)
- Hourly routes (gif)

## 15.03 Meeting
- Demand: Daily difference going in and out at each time step
- Get model, predict demand, use demand to solve movement of bikes based on demand to minimise cost
- use network to train rnn which predicts it, takes in t and predicts t+1
- predict ahead of time what demand will be
- time step enough to give time to move bikes but short enough to be useful
- dynamic programming should be done
- minimise based on constraints
- bikes needed at each station, cost from node to node, minimise number moved to start (then move to distance)
- train neural network based on data from node to node - no weighting yet (will be distance)
- RNN ignore connections - source to destination
- how many bikes needed at each timestep

## Features
- Station Id
- Bikes In
- Bikes Out
- Time of Day
- Day of the week
- Rainfall
- Temperature

- Output: Number of bikes at next time step at each station
