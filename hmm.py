from typing import Pattern, Union, Tuple, List, Dict, Any

import numpy as np
import numpy.typing as npt

"""
Some type annotations
"""
Numeric = Union[float, int, np.number, None]


"""
Global list of parts of speech
"""
POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
       'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
num_pos = len(POS)

"""
Utility functions for reading files and sentences
"""
def read_sentence(f):
    sentence = []
    while True:
        line = f.readline()
        if not line or line == '\n':
            return sentence
        line = line.strip()
        word, tag = line.split("\t", 1)
        sentence.append((word, tag))

def read_corpus(file):
    f = open(file, 'r', encoding='utf-8')
    sentences = []
    while True:
        sentence = read_sentence(f)
        if sentence == []:
            return sentences
        sentences.append(sentence)


"""
3.1: Supervised learning
Param: data is a list of sentences, each of which is a list of (word, POS) tuples
Return: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities} 
"""
def learn_model(data:List[List[Tuple[str]]]
                ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str,npt.NDArray]]:
    pos_counts = np.zeros(num_pos)
    transition_counts = np.zeros((num_pos, num_pos))
    obs_counts = {}
    for sentence in data:
        for i in range(len(sentence)):
            word = sentence[i][0]
            pos = POS.index(sentence[i][1])
            pos_counts[pos] += 1
            if i < len(sentence) - 1:
                transition_counts[POS.index(sentence[i + 1][1]), pos] += 1
            if word not in obs_counts:
                obs_counts[word] = np.zeros(num_pos)
            obs_counts[word][pos] += 1
    X0 = pos_counts / np.sum(pos_counts)
    Tprob = transition_counts / np.sum(transition_counts, axis=0)
    Oprob = {}
    for word, counts in obs_counts.items():
        Oprob[word] = np.divide(counts, pos_counts)
    return X0, Tprob, Oprob


"""
3.2: Viterbi forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: m, 1D array; pointers, 2D array
"""

def viterbi_forward(X0:npt.NDArray,
                    Tprob:npt.NDArray,
                    Oprob:Dict[str,npt.NDArray],
                    obs:List[str]
                    ) -> Tuple[npt.NDArray, npt.NDArray]:
    m = X0
    pointers = []
    for o in obs:
        mprime = np.max(Tprob * m, axis=1)
        tags = np.argmax(Tprob * m, axis=1)
        if o in Oprob:
            m = Oprob[o] * mprime
        else:
            m = mprime
        pointers.append(tags)
    return m, pointers

"""
3.2: Viterbi backward algorithm
Param: m, 1D array; pointers, 2D array
Return: List of most likely POS (strings)
"""
def viterbi_backward(m:npt.NDArray,
                     pointers:npt.NDArray
                     ) -> List[str]:
    tag = np.argmax(m)
    sequence = [POS[tag]]
    for i in range(len(pointers) - 1):
        tag = (pointers[-i - 1])[tag]
        sequence.insert(0, POS[tag])
    return sequence


"""
3.3: Evaluate Viterbi by predicting on data set and returning accuracy rate
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of (word,POS) pairs
Return: Prediction accuracy rate
"""
def evaluate_viterbi(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     data:List[List[Tuple[str]]]
                     ) -> float:
    correct = 0
    total = 0
    for sentence in data:
        word_list = []
        pos_list = []
        for i in range(len(sentence)):
            word = sentence[i][0]
            pos = sentence[i][1]
            word_list.append(word)
            pos_list.append(pos)
        m, pointers = viterbi_forward(X0, Tprob, Oprob, word_list)
        sequence = viterbi_backward(m, pointers)
        for p in range(len(pos_list)):
            total += 1
            if pos_list[p] == sequence[p]:
                correct += 1
    accuracy = correct / total
    return accuracy


"""
3.4: Forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: P(XT, e_1:T)
"""
def forward(X0:npt.NDArray,
            Tprob:npt.NDArray,
            Oprob:Dict[str,npt.NDArray],
            obs:List[str]
            ) -> npt.NDArray:
    alpha = X0
    for o in obs:
        alpha_prime = Tprob @ alpha
        if o in Oprob:
            alpha = np.multiply(Oprob[o], alpha_prime)
        else:
            alpha = alpha_prime
    return alpha

"""
3.4: Backward algorithm
Param: Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(e_k+1:T | Xk)
"""

def backward(Tprob:npt.NDArray,
             Oprob:Dict[str,npt.NDArray],
             obs:List[str],
             k:int
             ) -> npt.NDArray:
    beta = np.ones(num_pos, dtype=int)
    for o in obs:
        if o in Oprob:
            beta_prime = np.multiply(Oprob[o], beta)
        else:
            beta_prime = beta
        beta = Tprob.transpose() @ beta_prime
        k -= 1
        if k < 0:
            break
    return beta

"""
3.4: Forward-backward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(Xk | e_1:T)
"""
def forward_backward(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     obs:List[str],
                     k:int
                     ) -> npt.NDArray:
    fb = np.zeros(num_pos)
    alpha = forward(X0,Tprob, Oprob, obs[:k+1])
    beta = backward(Tprob, Oprob, obs, k)
    total = np.zeros(num_pos)
    for i in range(num_pos):
        fb[i] = alpha[i] * beta[i]
        total += fb[i]
    norm_fb = fb / total
    return norm_fb


"""
3.5: Expected observation probabilities given data sequence
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of words
Return: New Oprob, dictionary {word:probabilities}
"""
def expected_emissions(X0:npt.NDArray,
                       Tprob:npt.NDArray,
                       Oprob:Dict[str,npt.NDArray],
                       data:List[List[str]]
                       ) -> Dict[str,npt.NDArray]:
    new_obs = {}
    total = np.zeros(num_pos)
    for sentence in data:
        word_list = []
        gamma_list = []
        for i in range(len(sentence)):
            word = sentence[i]
            word_list.append(word)
        for i in range(len(word_list)):
            gamma = forward_backward(X0, Tprob, Oprob, word_list, i)
            gamma_list.append(gamma)
        np.seterr(invalid='ignore')
        for i in range(len(word_list)):
            if word_list[i] not in new_obs:
                new_obs[word_list[i]] = np.divide(gamma_list[i], sum(gamma_list))
            else:
                new_obs[word_list[i]] += np.divide(gamma_list[i], sum(gamma_list))
            total += np.divide(gamma_list[i], sum(gamma_list))
    normalized_obs = {}
    for w, prob in new_obs.items():
        normalized_obs[w] = new_obs[w] / total
    return normalized_obs


if __name__ == "__main__":

    # Run below for 3.3
    train = read_corpus('train.upos.tsv')
    test = read_corpus('test.upos.tsv')
    X0, T, O = learn_model(train)
    print("Train accuracy:", evaluate_viterbi(X0, T, O, train))
    print("Test accuracy:", evaluate_viterbi(X0, T, O, test))


# Run below for 3.5
    obs = [[pair[0] for pair in sentence] for sentence in [test[0]]]
    Onew = expected_emissions(X0, T, O, obs)
    print(Onew)


