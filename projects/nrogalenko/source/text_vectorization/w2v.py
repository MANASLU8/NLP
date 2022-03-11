from gensim.models import Word2Vec
from .vectorize import tokenize_with_filter


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
    model = Word2Vec(sentences=training_list, window=5, min_count=1, workers=4)
    print(model)
    # save model
    model.save('../assets/w2v-test-model.bin')


def use_model():
    # load model
    model = Word2Vec.load('../assets/w2v-test-model.bin')
    print(model)
    print(model.wv.get_vector('letter'))


'''train_df = pd.read_csv('../assets/test.csv', header=None, names=["Label", "News header", "News text"])
train_df["Text"] = train_df["News header"] + ". " + train_df["News text"]
train_df.drop(columns=["News header", "News text"], axis=1, inplace=True)
train_df["Text"] = train_df["Text"].apply(tokenize_with_filter)
train_df.to_csv("../assets/a")'''
