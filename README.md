# Vote predicting neural networks

```
pipenv install
pipenv run python main.py
```

This codebase builds Neural Network Models for Parliamentarians of the European Union to predict votes on future legislation. The model calculates a parliamentarians vote on a certain issue using the position of relevant organizations as input. The trained learning models achieve in average an accuracy of 92.5% for current Parliamentarians of the EU.

Neural networks are a machine learning technique for supervised learning. A network is trained with labeled samples. Each sample trains the neurons in the network to produce an output as close as possible to its label. It does so by minimizing the cost, the difference between the label value and the calculated value. Each training iteration reduces this cost by updating the network parameters. The network is trained until every new iteration does not decrease the cost any further.

Training and testing the networks is based on votes of over 3600 present and past EU Parliamentarians, 120 Lobbyists and Organizations and their position on ~90 bills and legislative proposals between 2013 and 2018. Each Neural Network is trained for a specific EU Parliamentarian (MEP) using its votes as labels on samples. Each sample represents the Positions of Organization on Bills. The dataset has been gathered by [politix.io](https://politix.io).

# Network Architecture

```
labels = Parliamentarian vote (For/Against/Abstain) on a EU bill
samples = Positions of Organizations and Lobbyists towards bills.
```

one-vs-all classification for Parliamentarian `P`.

- number of classes `K=3` (for/against/abstain)
- number of features `nx`: organizations holding votes on bills with `P`
- 13 hidden layers of size nx activated by ReLU function.
- Ouput layer activiated with by `Sigmoid` function.
- Between 20 to 33 training samples per Parliamentarian `P`.
- Training set `y_train` holds 70% of labeled samples. Testing set holds 30% of labeled samples.

# Hyper paramters and optimization
```
Number of training iterations
2000

Number of hidden layers
13

Number of Features
between 40 and 100 per Parliamentarian
```

# Prediction Accuracy

The accuracy for Parliamentarians with 33 votes in the dataset is `94.3%`, 32 votes or more is `92.5%` for the given testing set.

# Credits

This project uses utility methods from [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) taught by Andrew Ng. Its based on the dataset of [politix.io](https://politix.io).

# Context
In 2013 my friends and I co-founded [politix.io](https://politix.io). The project aims to give a new angle on how we look at a governments work. We believe correlating Organizations, Parliamentarians and Citizens through their vote on legislation allows to understand sentiment clusters beyond official coalition. The Position of Organizations on legislation guides Parlamentarians in their decisions. While this has been a common instinct, politix.io researches those correlations. The project was funded by [Advocate Europe](https://advocate-europe.eu/) in 2016.
