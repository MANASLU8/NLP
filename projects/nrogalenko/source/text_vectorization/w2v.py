from gensim.models import Word2Vec
from nltk.corpus import stopwords
import numpy as np
from .vectorize import tokenize_with_filter, get_single_word_tf_idf
from .cosine import cosine_distance
from .reduce import pca_reduce
from source.text_annotation.tokenizer import tokenize


def w2v_train(training_file, window, vector_size, min_fr, resulting_path):
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
    model = Word2Vec(sentences=training_list, window=window, min_count=min_fr, workers=4, vector_size=vector_size)
    print(model)
    # save model
    model.save(resulting_path)


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


def create_w2v_doc_embedding(model, text, dict_file, matrix_file):
    sentences_list = []
    averaged_sentences = []
    # filter tokens
    eng_stopwords = stopwords.words("english")
    all_tokens_list = [token_info.token.lower() for token_info in tokenize(text)
                       if (token_info.token_tag != "whitespace" and token_info.token_tag != "quotation"
                           and token_info.token != "undefined token type" and token_info.token.lower() not in eng_stopwords)]
    current_sentence_tokens = []
    # get sentences with tokens weighted representation
    for token in all_tokens_list:
        if token in ['.', '!', '?', '...']:
            # end of sentence
            if current_sentence_tokens:
                sentences_list.append(current_sentence_tokens)
            current_sentence_tokens = []
        else:
            if token in list(model.wv.index_to_key):
                # get vector weighted representation
                tf_idf = get_single_word_tf_idf(token, dict_file, matrix_file)
                if tf_idf is not None:
                    weighted_vector = model.wv.get_vector(token) * tf_idf
                    current_sentence_tokens.append(weighted_vector)
    if not sentences_list:
        sentences_list.append(current_sentence_tokens)
    # get average sentences vectors
    for sentence in sentences_list:
        averaged_sentences.append(np.average(sentence, axis=0))
    document_vector = np.average(averaged_sentences, axis=0)
    return document_vector


def create_doc_embeddings_file(model_path, file_to_process,  dict_file, matrix_file, output_file):
    # load model
    model = Word2Vec.load(model_path)
    f_out = open(output_file, "w+")
    lines_counter = 1
    with open(file_to_process) as f_in:
        line = f_in.readline()
        while line:
            document_text = line.split('","')[1] + ". " + (line.split('","')[2])[:-1]
            document_vector = create_w2v_doc_embedding(model, document_text, dict_file, matrix_file)
            f_out.write(str(lines_counter) + "\t" + "\t".join(str(component) for component in document_vector))
            f_out.write("\n")
            print(lines_counter)
            lines_counter += 1
            line = f_in.readline()
    f_out.close()


