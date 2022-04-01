from lab2.levenst import *


def fixtypo(d, word, algo="Hirschberg", percent=50):
    min_score = len(word)
    to_fix = ""
    for d_word in d:
        if algo == 'Hirschberg':
            dist = lev_hirsch(d_word[0], word)
        else:
            dist = lev_vag_fish(d_word[0], word)
        # print("distance: ", dist)
        if dist < min_score:
            min_score = dist
            to_fix = d_word[0]
    # print(word, to_fix, min_score)
    if min_score == 0:
        return word, False
    if min_score/len(word) >= percent/100:
        # print('cannot fix', word, min_score)
        return word, True
    return to_fix, True
