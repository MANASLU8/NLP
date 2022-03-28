from gensim.models import Word2Vec
from lab1.reader import read_from_file
from lab3.corpus import corpus
from lab3.cos_dist import cos_dist, plot
from lab3.vectorize import resize
# Реализовать метод, позволяющий векторизовать произвольный текст с использованием
# нейронных сетей (предлагается использовать стандартную реализацию модели w2v или glove).
# Выбранную модель необходимо обучить на обучающей выборке.


def neur_train(train_path, res_path, min_count, size=100, workers=4):
    train = []
    lines = read_from_file(train_path)
    for line in lines:
        tokens = corpus(line)[0]
        # print(tokens)
        train.append(tokens)
    # min_count получаем из percent в main
    # print("\n\n\n", train)
    w2v = Word2Vec(sentences=train, min_count=min_count, vector_size=size, workers=workers)
    print(w2v)

    w2v.save(str(res_path))


def neur4tests(mod_path, token, group1, group2, group3):
    w2v = Word2Vec.load(mod_path)
    # print(w2v)
    tokens = [token]
    print(token)
    print("Похожие:")
    for t in group1:
        print(t, cos_dist(w2v.wv.get_vector(token), w2v.wv.get_vector(t)))
        tokens.append(t)
    print("Предметная область:")
    for t in group2:
        print(t, cos_dist(w2v.wv.get_vector(token), w2v.wv.get_vector(t)))
        tokens.append(t)
    print("Совершенно другие:")
    for t in group3:
        print(t, cos_dist(w2v.wv.get_vector(token), w2v.wv.get_vector(t)))
        tokens.append(t)

    vectors = []
    for t in tokens:
        vectors.append(w2v.wv.get_vector(t))
    compressed = resize(vectors, 2)
    return compressed, tokens


def compare_tokens(mod_path):
    tokens = ["japan", "july"]
    group1 = [["russia", "iran", "korea"], ["june", "month", "aug"]]
    group2 = [["mauritian", "indians", "asian"], ["week", "thursday", "year"]]
    group3 = [["growth", "cost", "hand"], ["growth", "cost", "hand"]]
    for i in range(len(tokens)):
        resized, whole = neur4tests(mod_path, tokens[i], group1[i], group2[i], group3[i])
        plot(tokens[i], resized, whole)
