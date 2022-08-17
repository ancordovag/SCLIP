from experiment import reciprocal_rank
import numpy as np


def random_classifier(batch_size, epochs):
    MRR = []
    for e in range(epochs):
        score = []
        probs = np.random.random((batch_size, batch_size))  # compute fake similarities -> go random!
        for i in range(batch_size):
            score.append(reciprocal_rank(probs[i], probs[i][0]))  # pretend GT is always in the first position
        MRR.append(np.mean(score))
    return np.mean(MRR)


if __name__ == '__main__':
    epochs = 1000
    batch_size = 64
    print("running Random Classifier with batch size {} for {} epoches and averaging.".format(batch_size, epochs))
    score = random_classifier(batch_size, epochs)
    print("average MRR score: {}".format(score))
