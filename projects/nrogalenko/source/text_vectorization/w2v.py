from gensim.models import Word2Vec
from .vectorize import tokenize_with_filter
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


def use_w2v_model(model_path, token_to_test, similar_tokens, same_field_tokens, other_tokens):
    # load model
    model = Word2Vec.load(model_path)
    print(model)
    all_tokens = []
    print("Token: " + token_to_test)
    all_tokens.append(token_to_test)
    print("Similar tokens:")
    for similar_token in similar_tokens:
        all_tokens.append(similar_token)
        print("\t" + similar_token + ": " + str(cosine_distance(model.wv.get_vector(token_to_test), model.wv.get_vector(similar_token))))
    print("Same field tokens:")
    for same_field_token in same_field_tokens:
        all_tokens.append(same_field_token)
        print("\t" + same_field_token + ": " + str(cosine_distance(model.wv.get_vector(token_to_test), model.wv.get_vector(same_field_token))))
    print("Other tokens:")
    for other_token in other_tokens:
        all_tokens.append(other_token)
        print("\t" + other_token + ": " + str(cosine_distance(model.wv.get_vector(token_to_test), model.wv.get_vector(other_token))))
    # fit a 2d PCA model to the vectors
    x = [model.wv.get_vector(token) for token in all_tokens]
    result = pca_reduce(x, 2)
    return result, all_tokens

