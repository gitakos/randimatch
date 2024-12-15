import os
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

def read_texts_from_folder(folder_path):
    """Beolvassa a megadott mappában található .txt fájlokat."""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                texts[filename] = file.read()
    return texts

def preprocess_text(text):
    """Előkészíti a szöveget tokenizálással és stop-szavak eltávolításával."""
    return preprocess_string(remove_stopwords(text.lower()))

def extract_features(text, model=None, reference_topic=None):
    """Kinyeri a szöveg jellemző tulajdonságait."""
    tokens = preprocess_text(text)
    word_count = len(tokens)
    unique_words = len(set(tokens))
    average_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    long_words_ratio = sum(1 for word in tokens if len(word) >= 7) / max(word_count, 1)
    syllables_per_word = np.mean([count_syllables(word) for word in tokens]) if tokens else 0
    vocabulary_density = unique_words / max(word_count, 1)

    # Karakterek gyakorisága (pl. 'e', 'a')
    char_counts = Counter(text)
    common_chars = [char_counts.get(c, 0) for c in 'ea']

    # Szóhasználati jellemző: leggyakoribb szavak
    word_frequencies = Counter(tokens)
    top_words = [word for word, _ in word_frequencies.most_common(5)]

    # Word2Vec alapú jellemzők
    word_vectors = [model[word] for word in tokens if model and word in model]
    avg_vector_length = np.mean([np.linalg.norm(vec) for vec in word_vectors]) if word_vectors else 0

    if reference_topic and model:
        topic_vector = np.mean([model[word] for word in reference_topic if word in model], axis=0)
        text_vector = np.mean(word_vectors, axis=0) if word_vectors else None
        topic_similarity = cosine_similarity([topic_vector], [text_vector])[0][0] if text_vector is not None else 0
    else:
        topic_similarity = 0

    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "average_word_length": average_word_length,
        "sentence_count": sentence_count,
        "long_words_ratio": long_words_ratio,
        "syllables_per_word": syllables_per_word,
        "vocabulary_density": vocabulary_density,
        "common_chars": common_chars,
        "avg_vector_length": avg_vector_length,
        "topic_similarity": topic_similarity,
        "top_words": top_words
    }

def count_syllables(word):
    """Becsüli egy szó szótagjainak számát."""
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(1, count)



def calculate_similarity(features1, features2, text1_tokens, text2_tokens, model):
    """Kiszámítja a két szöveg közötti hasonlóságot a jellemzők és Word2Vec vektorok alapján."""
    # Tulajdonság alapú hasonlóság
    numeric_features1 = {k: v for k, v in features1.items() if not isinstance(v, list)}
    numeric_features2 = {k: v for k, v in features2.items() if not isinstance(v, list)}
    feature_similarity = max(0, 1 - np.linalg.norm(
        np.array(list(numeric_features1.values())) - np.array(list(numeric_features2.values()))
    ))

    # Word2Vec alapú szöveg-hasonlóság
    text1_vectors = [model[word] for word in text1_tokens if word in model.key_to_index]
    text2_vectors = [model[word] for word in text2_tokens if word in model.key_to_index]

    if text1_vectors and text2_vectors:
        text1_mean_vector = np.mean(text1_vectors, axis=0)
        text2_mean_vector = np.mean(text2_vectors, axis=0)
        word2vec_similarity = cosine_similarity([text1_mean_vector], [text2_mean_vector])[0][0]
    else:
        word2vec_similarity = 0

    # Szóhasználat alapú hasonlóság
    common_words = set(features1["top_words"]) & set(features2["top_words"])
    usage_similarity = len(common_words) / max(len(features1["top_words"]), len(features2["top_words"]), 1)

    # Összetett hasonlóság (kombinált értékelés)
    return 0.4 * feature_similarity + 0.3 * usage_similarity + 0.3 * word2vec_similarity





def pair_texts(texts1, texts2, model):
    """Párosítja a két mappa szövegeit tulajdonságok és Word2Vec hasonlóság alapján."""
    pairs = []
    for name1, text1 in texts1.items():
        best_match = None
        best_score = -1
        features1 = extract_features(text1, model)
        text1_tokens = preprocess_text(text1)
        for name2, text2 in texts2.items():
            features2 = extract_features(text2, model)
            text2_tokens = preprocess_text(text2)
            similarity = calculate_similarity(features1, features2, text1_tokens, text2_tokens, model)
            if similarity > best_score:
                best_match = name2
                best_score = similarity
        pairs.append((name1, best_match, best_score))
    return pairs




def load_pretrained_word2vec_model(path):
    """Betölt egy előre tanított Word2Vec modellt."""
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

def main(folder1, folder2, model_path):
    # Beolvasás
    texts1 = read_texts_from_folder(folder1)
    texts2 = read_texts_from_folder(folder2)

    # Word2Vec modell betöltése
    model = load_pretrained_word2vec_model(model_path)

    # Szövegek párosítása
    pairs = pair_texts(texts1, texts2, model)

    # Eredmények kiírása
    print("Párosítások és hasonlósági pontszámok:")
    for name1, name2, score in pairs:
        print(f"{name1} <--> {name2} (Hasonlóság: {score:.2f})")

def pair_texts_bidirectional_unique(texts1, texts2, model):
    """Párosítja a két mappa szövegeit mindkét irányban, és megtalálja a legjobb párokat."""
    best_pairs_from_first = {}
    best_pairs_from_second = {}

    # Első mappa szövegeihez legjobb párok a másodikból
    for name1, text1 in texts1.items():
        best_match = None
        best_score = -1
        features1 = extract_features(text1, model)
        text1_tokens = preprocess_text(text1)
        for name2, text2 in texts2.items():
            features2 = extract_features(text2, model)
            text2_tokens = preprocess_text(text2)
            similarity = calculate_similarity(features1, features2, text1_tokens, text2_tokens, model)
            if similarity > best_score:
                best_match = name2
                best_score = similarity
        best_pairs_from_first[name1] = (best_match, best_score)

    # Második mappa szövegeihez legjobb párok az elsőből
    for name2, text2 in texts2.items():
        best_match = None
        best_score = -1
        features2 = extract_features(text2, model)
        text2_tokens = preprocess_text(text2)
        for name1, text1 in texts1.items():
            features1 = extract_features(text1, model)
            text1_tokens = preprocess_text(text1)
            similarity = calculate_similarity(features2, features1, text2_tokens, text1_tokens, model)
            if similarity > best_score:
                best_match = name1
                best_score = similarity
        best_pairs_from_second[name2] = (best_match, best_score)

    # Kölcsönösen legjobb párok meghatározása
    final_pairs = []
    for name1, (best_match1, score1) in best_pairs_from_first.items():
        reciprocal_match, score2 = best_pairs_from_second.get(best_match1, (None, 0))
        if reciprocal_match == name1:
            final_pairs.append((name1, best_match1, score1))

    # A legjobb párok kiírása, mindkét irányból
    all_pairs = []
    for name1, (best_match1, score1) in best_pairs_from_first.items():
        all_pairs.append((name1, best_match1, score1))
    
    for name2, (best_match2, score2) in best_pairs_from_second.items():
        all_pairs.append((best_match2, name2, score2))

    return all_pairs

def print_all_pair_results(all_pairs):
    """Kiírja az összes párosítást és azok hasonlósági pontszámát."""
    print("\nPárosítások és hasonlósági pontszámok:")
    for name1, name2, score in all_pairs:
        print(f"{name1} <--> {name2} (Hasonlóság: {score:.2f})")



if __name__ == "__main__":
    folder1 = "naplo1"
    folder2 = "naplo2"
    model_path = "GoogleNews-vectors-negative300.bin.gz"

    # Szövegek beolvasása
    texts1 = read_texts_from_folder(folder1)
    texts2 = read_texts_from_folder(folder2)

    # Word2Vec modell betöltése
    model = load_pretrained_word2vec_model(model_path)

    # Szövegek párosítása kölcsönösen a legjobb párokkal
    all_pairs = pair_texts_bidirectional_unique(texts1, texts2, model)

    # Eredmények kiírása
    print_all_pair_results(all_pairs)
