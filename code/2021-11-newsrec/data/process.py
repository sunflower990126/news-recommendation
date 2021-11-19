import os
import json
import random
import numpy as np
from nltk.tokenize import RegexpTokenizer

random.seed(42)

def read_clickhistory(path, filename):
    """Read click history file

    Args:
        path (str): Folder path
        filename (str): Filename
    Returns:
        list, dict:
        - A list of user session with user_id, clicks, positive and negative interactions.
        - A dictionary with user_id click history.
    """
    userid_history = {}
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split("\t")
        clicks = click.split(" ")
        pos = []
        neg = []
        imps = imps.split(" ")
        for imp in imps:
            if imp.split("-")[1] == "1":
                pos.append(imp.split("-")[0])
            else:
                neg.append(imp.split("-")[0])
        userid_history[userid] = clicks
        sessions.append([userid, clicks, pos, neg])
    return sessions, userid_history


def _newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def get_train_input(session, train_file_path, npratio=4):
    """Generate train file.

    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        train_file_path (str): Path to file.
        npration (int): Ratio for negative sampling.
    """
    fp_train = open(train_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        sess = session[sess_id]
        userid, _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = _newsample(negs, npratio)
            fp_train.write("1 " + "train_" + userid + " " + pos + "\n")
            for neg_ins in neg:
                fp_train.write("0 " + "train_" + userid + " " + neg_ins + "\n")
    fp_train.close()
    if os.path.isfile(train_file_path):
        print(f"Train file {train_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {train_file_path}")


def get_valid_input(session, valid_file_path):
    """Generate validation file.
    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        valid_file_path (str): Path to file.
    """
    fp_valid = open(valid_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        userid, _, poss, negs = session[sess_id]
        for i in range(len(poss)):
            fp_valid.write(
                "1 " + "valid_" + userid + " " +
                poss[i] + "%" + str(sess_id) + "\n"
            )
        for i in range(len(negs)):
            fp_valid.write(
                "0 " + "valid_" + userid + " " +
                negs[i] + "%" + str(sess_id) + "\n"
            )
    fp_valid.close()
    if os.path.isfile(valid_file_path):
        print(f"Validation file {valid_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {valid_file_path}")


def get_user_history(train_history, valid_history, test_history, user_history_path):
    """Generate user history file.
    Args:
        train_history (list): Train history.
        valid_history (list): Validation history
        user_history_path (str): Path to file.
    """
    fp_user_history = open(user_history_path, "w", encoding="utf-8")
    for userid in train_history:
        fp_user_history.write(
            "train_" + userid + " " + ",".join(train_history[userid]) + "\n"
        )
    for userid in valid_history:
        fp_user_history.write(
            "valid_" + userid + " " + ",".join(valid_history[userid]) + "\n"
        )
    if test_history is not None:
        for userid in test_history:
            fp_user_history.write(
                "valid_" + userid + " " + ",".join(test_history[userid]) + "\n"
            )
    fp_user_history.close()
    if os.path.isfile(user_history_path):
        print(f"User history file {user_history_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {user_history_path}")


def _read_news(filepath, news_words, news_entities, tokenizer):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.strip("\n").split("\t")
        news_words[splitted[0]] = tokenizer.tokenize(splitted[3].lower())
        news_entities[splitted[0]] = []
        for entity in json.loads(splitted[6]):
            news_entities[splitted[0]].append(
                (entity["SurfaceForms"], entity["WikidataId"])
            )
    return news_words, news_entities


def get_words_and_entities(train_news, valid_news, test_news):
    """Load words and entities
    Args:
        train_news (str): News train file.
        valid_news (str): News validation file.
    Returns:
        dict, dict: Words and entities dictionaries.
    """
    news_words = {}
    news_entities = {}
    tokenizer = RegexpTokenizer(r"\w+")
    news_words, news_entities = _read_news(
        train_news, news_words, news_entities, tokenizer
    )
    news_words, news_entities = _read_news(
        valid_news, news_words, news_entities, tokenizer
    )
    news_words, news_entities = _read_news(
        test_news, news_words, news_entities, tokenizer
    )
    return news_words, news_entities


def generate_embeddings(
    data_path,
    news_words,
    news_entities,
    train_entities,
    valid_entities,
    test_entities,
    max_sentence=10,
    word_embedding_dim=100,
):
    """Generate embeddings.
    Args:
        data_path (str): Data path.
        news_words (dict): News word dictionary.
        news_entities (dict): News entity dictionary.
        train_entities (str): Train entity file.
        valid_entities (str): Validation entity file.
        max_sentence (int): Max sentence size.
        word_embedding_dim (int): Word embedding dimension.
    Returns:
        str, str, str: File paths to news, word and entity embeddings.
    """
    embedding_dimensions = [50, 100, 200, 300]
    if word_embedding_dim not in embedding_dimensions:
        raise ValueError(
            f"Wrong embedding dimension, available options are {embedding_dimensions}"
        )

    print("Downloading glove...")
    glove_path = os.path.join(data_path, "glove")

    word_set = set()
    word_embedding_dict = {}
    entity_embedding_dict = {}

    print(
        f"Loading glove with embedding dimension {word_embedding_dim}...")
    glove_file = "glove.6B." + str(word_embedding_dim) + "d.txt"
    fp_pretrain_vec = open(os.path.join(
        glove_path, glove_file), "r", encoding="utf-8")
    for line in fp_pretrain_vec:
        linesplit = line.split(" ")
        word_set.add(linesplit[0])
        word_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:])))
    fp_pretrain_vec.close()

    print("Reading train entities...")
    fp_entity_vec_train = open(train_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_train:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_train.close()

    print("Reading valid entities...")
    fp_entity_vec_valid = open(valid_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_valid:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_valid.close()

    print("Reading test entities...")
    fp_entity_vec_test = open(test_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_test:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_test.close()

    print("Generating word and entity indexes...")
    word_dict = {}
    word_index = 1
    news_word_string_dict = {}
    news_entity_string_dict = {}
    entity2index = {}
    entity_index = 1
    for doc_id in news_words:
        news_word_string_dict[doc_id] = [0 for n in range(max_sentence)]
        news_entity_string_dict[doc_id] = [0 for n in range(max_sentence)]
        surfaceform_entityids = news_entities[doc_id]
        for item in surfaceform_entityids:
            if item[1] not in entity2index and item[1] in entity_embedding_dict:
                entity2index[item[1]] = entity_index
                entity_index = entity_index + 1
        for i in range(len(news_words[doc_id])):
            if news_words[doc_id][i] in word_embedding_dict:
                if news_words[doc_id][i] not in word_dict:
                    word_dict[news_words[doc_id][i]] = word_index
                    word_index = word_index + 1
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                else:
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                for item in surfaceform_entityids:
                    for surface in item[0]:
                        for surface_word in surface.split(" "):
                            if news_words[doc_id][i] == surface_word.lower():
                                if item[1] in entity_embedding_dict:
                                    news_entity_string_dict[doc_id][i] = entity2index[
                                        item[1]
                                    ]
            if i == max_sentence - 1:
                break

    print("Generating word embeddings...")
    word_embeddings = np.zeros([word_index, word_embedding_dim])
    for word in word_dict:
        word_embeddings[word_dict[word]] = word_embedding_dict[word]

    print("Generating entity embeddings...")
    entity_embeddings = np.zeros([entity_index, word_embedding_dim])
    for entity in entity2index:
        entity_embeddings[entity2index[entity]] = entity_embedding_dict[entity]

    news_feature_path = os.path.join(data_path, "doc_feature.txt")
    print(f"Saving word and entity features in {news_feature_path}")
    fp_doc_string = open(news_feature_path, "w", encoding="utf-8")
    for doc_id in news_word_string_dict:
        fp_doc_string.write(
            doc_id
            + " "
            + ",".join(list(map(str, news_word_string_dict[doc_id])))
            + " "
            + ",".join(list(map(str, news_entity_string_dict[doc_id])))
            + "\n"
        )

    word_embeddings_path = os.path.join(
        data_path, "word_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    print(f"Saving word embeddings in {word_embeddings_path}")
    np.save(word_embeddings_path, word_embeddings)

    entity_embeddings_path = os.path.join(
        data_path, "entity_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    print(f"Saving word embeddings in {entity_embeddings_path}")
    np.save(entity_embeddings_path, entity_embeddings)

    return news_feature_path, word_embeddings_path, entity_embeddings_path



if __name__ == "__main__":
    have_test = False
    data_path = "../../data/MINDsmall"
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    train_file = os.path.join(train_path, "train.txt")
    valid_file = os.path.join(valid_path, "valid.txt")
    if have_test:
        test_path = os.path.join(data_path, "test")
        test_file = os.path.join(test_path, "test.txt")
    user_history_file = os.path.join(data_path, "user_history.txt")

    train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
    valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
    get_train_input(train_session, train_file)
    get_valid_input(valid_session, valid_file)
    if have_test:
        test_session, test_history = read_clickhistory(test_path, "behaviors.tsv")
        get_valid_input(test_session, test_file)
    else:
        test_history = None
    get_user_history(train_history, valid_history, test_history, user_history_file)

    train_news = os.path.join(train_path, "news.tsv")
    valid_news = os.path.join(valid_path, "news.tsv")
    if have_test:
        test_news = os.path.join(test_path, "news.tsv")
    else:
        test_news = None
    news_words, news_entities = get_words_and_entities(train_news, valid_news, test_news)

    train_entities = os.path.join(train_path, "entity_embedding.vec")
    valid_entities = os.path.join(valid_path, "entity_embedding.vec")
    if have_test:
        test_entities = os.path.join(test_path, "entity_embedding.vec")
    else:
        test_entities = None
    news_feature_file, word_embeddings_file, entity_embeddings_file = generate_embeddings(
        data_path,
        news_words,
        news_entities,
        train_entities,
        valid_entities,
        test_entities,
        max_sentence=10,
        word_embedding_dim=100,
    )
