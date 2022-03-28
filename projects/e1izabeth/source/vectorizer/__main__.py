from utils import create_plot, pca_reduce, cosine_distance
from vectorizer import TextVectorizer
from freq_dictionary import read_freq_dict
from w2v import apply_w2v_model, w2v_train


def apply_tfidf_model(tokens_entry, token_to_test, similar_tokens, same_field_tokens, other_tokens, vec_dimension):
    print("Tf-idf")
    tokens = [token_to_test]
    tokens = tokens + similar_tokens + same_field_tokens + other_tokens

    vectorizer = TextVectorizer(tokens_entry)
    x = [vectorizer.get_tf_idf_for_text(token) for token in tokens]
    reduced_embeddings = pca_reduce(x, vec_dimension)
    print("Cosine distance " + tokens[0])
    for i in range(1, len(tokens)):
        print("\t " + tokens[i] + ": " + str(cosine_distance(reduced_embeddings[0], reduced_embeddings[i])))
    return reduced_embeddings, tokens


def demonstrate_w2v_and_tfidf_models(model_path, tokens_entry, vec_dimension):
    initial_tokens = ["basketball", "company", "sudan",]
    similar_tokens = [["football", "soccer"], ["firm", "business", "agency"], ["california", "boston", "chicago"]]
    same_field_tokens = [["sport", "players"], ["bank", "economy"], ["north", "congress"]]
    other_tokens = [["yesterday", "fresh", "rules"], ["re-election", "press", "victory"], ["jail", "sentence", "conviction"]]
    for token in initial_tokens:
        reduced_embeddings, all_tokens = apply_w2v_model(model_path, token, similar_tokens[initial_tokens.index(token)],
            same_field_tokens[initial_tokens.index(token)], other_tokens[initial_tokens.index(token)], vec_dimension)
        create_plot("w2v - " + token, reduced_embeddings, all_tokens)

        reduced_embeddings, all_tokens = apply_tfidf_model(tokens_entry, token, similar_tokens[initial_tokens.index(token)],
            same_field_tokens[initial_tokens.index(token)], other_tokens[initial_tokens.index(token)], vec_dimension)
        create_plot("tf-idf - " + token, reduced_embeddings, all_tokens)


def main():
    vec_dict_file_path = "../assets/vec_dict"
    #tokens_entry = dict()
    #construct_entry_info(tokens_entry)
    #write_tokens_freq_dictionary(tokens_entry, vec_dict_file_path)
    tokens_entries = read_freq_dict(vec_dict_file_path)
    model_path = '../assets/w2v-train-model_5_100_5.bin'
    w2v_train('../assets/raw-dataset/train.csv', 10, 5, 100, model_path)
    #demonstrate_w2v_and_tfidf_models(model_path, tokens_entries, 2)
    vectorizer = TextVectorizer(tokens_entries)
    vectorizer.get_w2v_embeddings(model_path, "../assets/raw-dataset/train.csv", "../assets/annotated-corpus/train-embeddings.tsv")
    #print(tokens_entries)


if __name__ == "__main__":
    main()
