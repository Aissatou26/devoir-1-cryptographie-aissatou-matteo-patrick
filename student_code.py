from collections import Counter
import math
import random
from crypt_1 import chiffrer, chiffrer2, cut_string_into_pairs, gen_key, load_text_from_web


def decrypt(C):
    import math
    import random
    from collections import Counter

    # Définition des symboles : lettres de l'alphabet + espace
    symboles = list('abcdefghijklmnopqrstuvwxyz ')

    # Chargement et nettoyage du corpus à l'intérieur de la fonction
    urls = [
        "https://www.gutenberg.org/files/13846/13846-0.txt",
        "https://www.gutenberg.org/files/4650/4650-0.txt",
    ]
    corpus = ""
    for url in urls:
        corpus += load_text_from_web(url)
    corpus = nettoyer_texte(corpus, symboles)

    # Calculer les fréquences des symboles dans le corpus
    symbol_counts = Counter(corpus)
    total_symbols = sum(symbol_counts.values())
    symbol_freqs = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}

    # Diviser le cryptogramme en codes de 8 bits
    codes = [C[i:i + 8] for i in range(0, len(C), 8)]

    # Calculer les fréquences des codes
    code_counts = Counter(codes)
    total_codes = len(codes)
    code_freqs = {code: count / total_codes for code, count in code_counts.items()}

    # Établir un mapping initial basé sur les fréquences
    sorted_symbols = [symbol for symbol, _ in sorted(symbol_counts.items(), key=lambda item: item[1], reverse=True)]
    sorted_codes = [code for code, _ in sorted(code_counts.items(), key=lambda item: item[1], reverse=True)]
    initial_mapping = dict(zip(sorted_codes, sorted_symbols))

    # Fonction pour déchiffrer avec un mapping donné
    def decrypt_with_mapping(codes, mapping):
        decrypted_symbols = [mapping.get(code, '') for code in codes]
        plaintext = ''.join(decrypted_symbols)
        return plaintext

    # Construire un modèle de langue basé sur les trigrammes
    def build_ngram_model(corpus_text, n=3):
        ngram_counts = Counter()
        total_ngrams = 0
        for i in range(len(corpus_text) - n + 1):
            ngram = corpus_text[i:i + n]
            ngram_counts[ngram] += 1
            total_ngrams += 1
        return ngram_counts, total_ngrams

    ngram_counts, total_ngrams = build_ngram_model(corpus, n=3)

    # Fonction pour évaluer le texte déchiffré
    def score_text(text, ngram_counts, total_ngrams):
        score = 0
        n = 3
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            if ngram in ngram_counts:
                score += math.log(ngram_counts[ngram] / total_ngrams)
            else:
                score += math.log(1e-6)  # Pénalité pour les n-grammes inconnus
        return score

    # Algorithme de recuit simulé
    def simulated_annealing(codes, initial_mapping, ngram_counts, total_ngrams, iterations=50000, temperature=1.0, cooling_rate=0.00001):
        mapping = initial_mapping.copy()
        best_score = score_text(decrypt_with_mapping(codes, mapping), ngram_counts, total_ngrams)
        best_mapping = mapping.copy()
        current_score = best_score
        for iteration in range(iterations):
            temperature *= (1 - cooling_rate)
            code1, code2 = random.sample(list(mapping.keys()), 2)
            # Échanger les symboles associés aux codes
            mapping[code1], mapping[code2] = mapping[code2], mapping[code1]
            decrypted_text = decrypt_with_mapping(codes, mapping)
            new_score = score_text(decrypted_text, ngram_counts, total_ngrams)
            delta_score = new_score - current_score
            if delta_score > 0 or random.uniform(0, 1) < math.exp(delta_score / temperature):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_mapping = mapping.copy()
                    # Optionnel : afficher les améliorations
                    # print(f"Amélioration à l'itération {iteration}: Score = {best_score}")
            else:
                # Revenir au mapping précédent
                mapping[code1], mapping[code2] = mapping[code2], mapping[code1]
        return decrypt_with_mapping(codes, best_mapping)

    # Appliquer le recuit simulé
    M = simulated_annealing(codes, initial_mapping, ngram_counts, total_ngrams)

    return M
 

symboles = list('abcdefghijklmnopqrstuvwxyz ')

def chiffrer_simple(M, K):
    C = ''.join(K[c] for c in M if c in K)
    return C
# Fonction pour nettoyer le texte
def nettoyer_texte(texte, symboles):
    texte = texte.lower()
    texte = ''.join(c for c in texte if c in symboles)
    return texte
# Chargement et nettoyage du corpus
urls = [
    "https://www.gutenberg.org/files/13846/13846-0.txt",
    "https://www.gutenberg.org/files/4650/4650-0.txt",
]
corpus = ""
for url in urls:
    corpus += load_text_from_web(url)
corpus = nettoyer_texte(corpus, symboles)

# Génération de la clé
K = gen_key(symboles)

# Sélection du message à chiffrer
M = nettoyer_texte(corpus[10000:10500], symboles)




# Chiffrement du message
C = chiffrer_simple(M, K)

# Déchiffrement du message sans connaître la clé K
message_dechiffre = decrypt(C)

# Affichage du résultat
print("Message déchiffré :")
print(message_dechiffre)

# Affichage du message original pour comparaison
print("\nMessage original :")
print(M)