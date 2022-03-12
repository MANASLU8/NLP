from gensim.models import Word2Vec
from .vectorize import tokenize_with_filter
from matplotlib import pyplot
from .cosine import cosine_distance
from .reduce import pca_reduce


def w2v_train(training_file):
    # prepare data
    lines_counter = 1
    training_list = []
    with open(training_file) as f:
        line = f.readline()
        while line:
            news_header_tokens_list = tokenize_with_filter(line.split('","')[1])
            news_text_tokens_list = tokenize_with_filter(line.split('","')[2])
            training_list.append(news_header_tokens_list)
            training_list.append(news_text_tokens_list)
            print(lines_counter)
            lines_counter += 1
            line = f.readline()
    f.close()
    # train
    print("Training started")
    model = Word2Vec(sentences=training_list, window=5, min_count=5, workers=4, vector_size=100)
    print(model)
    # save model
    model.save('../assets/w2v-train-model_100_5.bin')


def use_model():
    # load model
    model = Word2Vec.load('../assets/w2v-train-model_100_5.bin')
    print(model)
    chosen_tokens = ["football"]
    similar_tokens = [["basketball", "volleyball"]]
    same_field_tokens = [["sport", "game", "man"]]
    other_tokens = [["weekend", "bombing", "russia", "woman", "trump", "university"]]
    all_tokens = []
    for token in chosen_tokens:
        print("Token: " + token)
        all_tokens.append(token)
        print("Similar tokens:")
        for similar_token in similar_tokens[chosen_tokens.index(token)]:
            all_tokens.append(similar_token)
            print("\t" + similar_token + ": " + str(cosine_distance(model.wv.get_vector(token), model.wv.get_vector(similar_token))))
        print("Same field tokens:")
        for same_field_token in same_field_tokens[chosen_tokens.index(token)]:
            all_tokens.append(same_field_token)
            print("\t" + same_field_token + ": " + str(cosine_distance(model.wv.get_vector(token), model.wv.get_vector(same_field_token))))
        print("Other tokens:")
        for other_token in other_tokens[chosen_tokens.index(token)]:
            all_tokens.append(other_token)
            print("\t" + other_token + ": " + str(cosine_distance(model.wv.get_vector(token), model.wv.get_vector(other_token))))
    # print(model.wv.most_similar("football"))
    # fit a 2d PCA model to the vectors
    x = [model.wv.get_vector(token) for token in all_tokens]
    result = pca_reduce(x, 2)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = all_tokens
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

