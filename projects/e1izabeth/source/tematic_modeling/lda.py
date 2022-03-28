class LDA:
    def __init__(self, lda_model, model_path):
        self.lda_model = lda_model
        self.model_path = model_path

    def train(self, matrix):
        self.lda_model.fit(matrix)

    def write_perplexity_to_file(path, matrix, lda_model):
        perplexity_file = open(path, "w+")
        perplexity_file.write("iter: " + str(lda_model.max_iter) + "\t" + "components: " + str(lda_model.n_components) + "\t" + "perplexity: " + str(lda_model.perplexity(matrix)) + "\n")
        perplexity_file.close()

    def get_dt_matrix_and_save(self, path, dt_matrix):
        doc_topic = self.lda_model.transform(dt_matrix)
        dt_probability = [[] for _ in range(len(doc_topic[0]))]
        dt_file = open(path, "w+")
        for n, doc in enumerate(doc_topic):
            dt_string_builder = str(n + 1)
            for num, t_prob in enumerate(doc):
                dt_probability[num].append(t_prob)
                dt_string_builder += ("\t" + str(t_prob))
            dt_file.write(dt_string_builder + "\n")
        dt_file.close()
        return dt_probability

    def top_words_by_topic_to_file(self, path, vectorizer, dt_probability, top_number):
        top_words_file = open(path, 'w+')
        for i, topic in enumerate(self.lda_model.components_):
            top_words_file.write("Topic" + (i + 1) + "\n")
            top_words_file.write(" ".join([vectorizer.all_words[i] for i in topic.argsort()[:- top_number - 1:-1]]) + "\n")
            top_docs_indices = sorted(range(len(dt_probability[i])), key=lambda j: dt_probability[i][j])[-top_number:]
            for index in top_docs_indices:
                top_words_file.write(vectorizer.fnames[index])
            top_words_file.write("\n")
        top_words_file.close()
