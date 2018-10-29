# Vote predicting neural networks

```
pipenv install
pipenv run python main.py
```

This codebase builds Neural Network Models for Parliamentarians of the European Union (EU) predicting their votes on future legislation. The model calculates a parliamentarians vote on a certain legislation using the Position of relevant Organizations and Lobbyists as input. The trained learning models achieve in average an accuracy of 92.5% over current Parliamentarians of the EU.

Neural networks are a machine learning technique for supervised learning. A network is trained with labeled samples. Each sample trains the neurons in the network to produce an output as close as possible to its label. It does so by minimizing the cost, the difference between the label value and the calculated value. Each training iteration reduces this cost by updating the network parameters. The network is trained until new training iterations do not decrease the cost much further, i.e. the cost is `minimal`. This project uses multiclass classification, one-vs-all, with `K=3` classes: for, against, abstain.

Training and testing the networks is based on votes of over 3600 present and past EU Parliamentarians, 120 Lobbyists and Organizations and their position on ~90 bills and legislative proposals between 2013 and 2018. Each Neural Network is trained for a specific EU Parliamentarian (MEP) using its votes as labels on samples. Each sample represents the Positions of Organization on Bills.

The next Elections to the European Parliament are expected to be held in 23–26 May 2019.

### Prediction accuracy

The accuracy for Parliamentarians with 33 votes in the dataset is `94.3%`, 32 votes or more is `92.5%` for the given testset (30% of the dataset).

### Network architecture

- one-vs-all classification for Parliamentarian `P`.
- labels = Parliamentarian vote (For/Against/Abstain) on a EU bill
- features = Positions of Organizations and Lobbyists towards bills
- number of classes `K=3` (for/against/abstain)
- number of features `nx`: organizations holding votes on bills with `P`
- 13 hidden layers of size nx activated by `ReLU` function.
- Ouput layer activiated by `Sigmoid` function.
- Between 20 to 33 training samples per Parliamentarian `P`.
- Training set `y_train` holds 70% of labeled samples. Testing set holds 30% of labeled samples.

### Hyper parameter optimization

The best predictions are achieved by a network with `13` hidden layers predicting votes with an accuracy of `92.5%`. The number of features `nx` is between 40 and 100 for each Parliamentarian. Each hidden layer is of size `nx`.

![Alt text](graphs/votePredictionAccuracy.png?raw=true "Vote Prediction Accuracy")

### Credits

This project uses utility methods from [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) taught by Andrew Ng. The dataset has been gathered by the team of [politix.io](https://politix.io) which extracted Parliamentarian votes from [parltrack.eu](http://parltrack.euwiki.org/). :pray:

### Prelude
In 2013 my friends and I co-founded [politix.io](https://politix.io). The project aims to give a new angle on how we look at a governments work. We believe correlating Organizations, Parliamentarians and Citizens through their vote on legislation allows to understand sentiment clusters beyond official coalition. The Position of Organizations on legislation guides Parlamentarians in their decisions. While this has been a common instinct, politix.io researches those correlations. The project was funded by [Advocate Europe](https://advocate-europe.eu/) in 2016 :pray:, won the [European Citizenship Award](http://civic-forum.eu/the-european-citizenship-awards) 2016 and has been presented at [RE:PUBLICA 2017](https://archiv-17.re-publica.com/en/session/politix-eu-closing-feedback-loop-eu-politics) in Berlin :sparkles:.
