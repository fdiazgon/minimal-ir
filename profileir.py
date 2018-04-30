import math
import os
import pprint
import re
from collections import defaultdict

DOCUMENTS_DIR = 'corpus'
PROFILES_FILE = 'profiles'
DICTIONARY_FILE = 'dictionary'

DELIMITER = '#'
VALID_EXTENSIONS = ['.txt']

SCORE_THRESHOLD = 0.1
SCORE_MULTIPLIER = 5.


class Profile:
    """A class to represent the profile of an user."""

    def __init__(self, name, interests):
        self.name = name
        self.interests = set(interests)
        self.recommendations = dict()

    def add_recommendation(self, document, score):
        self.recommendations[document] = score

    def show_recommendations(self):
        print('=' * 42)
        print(self.name.center(42, ' '))
        print('=' * 42)
        print('Interests: {}'.format(' & '.join(self.interests)).center(42, ' '))
        print('=' * 42)
        print('Recommendation'.center(20, ' ') + '||' + 'Score'.center(20, ' '))
        print('=' * 42)
        for document in sorted(self.recommendations, key=self.recommendations.get, reverse=True):
            document_id = document[0:16]
            score = self.recommendations[document]
            print(document_id.center(20, ' ') + '||' + str(score).center(20, ' '))
            print('=' * 42)


class VectorSpace:
    """A class to represent the N dimensional vector space formed by the terms."""

    def __init__(self, terms):
        self.axes = terms

    def as_normalized_vector(self, point):
        """Given a point in the vector space (x1, x2, ..., xN), represent the vector whose origin is at the origin of
        the coordinate system and the head is at the given point. Return the head of the normalized vector.

        Args:
            point (dict[str, int]): head of the vector in this space. For example: { 'x1': a, 'x2': b, 'x3': c }

        Returns:
            tuple[int]: head of the normalized vector of origin 0.

        """
        head = [point.get(axis, 0) for axis in self.axes]
        mod = math.sqrt(sum([x ** 2 for x in head]))
        if mod != 0:
            return tuple(x / mod for x in head)
        else:
            return tuple(head)

    def cos(self, a, b):
        """Return the cosine of the angle between two vectors in this space.

        Args:
            a (tuple[int]): head of the first vector with origin 0.
            b (tuple[int]): head of the second vector with origin 0.

        Returns:
            float: cosine of the angle formed by the two vectors.

        """
        assert len(a) == len(b) == len(self.axes), 'Dimension mismatch. Try to call as_normalized_vector first'
        dot_prod = sum([a[i] * b[i] for i in range(len(b))])
        mod_a = math.sqrt(sum([a[i] ** 2 for i in range(len(a))]))
        mod_b = math.sqrt(sum([b[i] ** 2 for i in range(len(b))]))
        mod_prod = mod_a * mod_b
        if mod_prod == 0:
            return float(0)
        else:
            return float(dot_prod / (mod_a * mod_b))


def basic_stemming(word):
    """Reduce a word following the following rules:

    Rule
    SSESS -> SS
    IES   -> I
    SS    -> SS
    S     ->

    Args:
        word (str): the word to reduce.
    Returns:
        str: reduced version of the word (if possible).

    """
    if type(word) == str:
        if word.endswith('sses'):
            word = re.sub('sses$', 'ss', word)
        elif word.endswith('ies'):
            word = re.sub('ies$', 'i', word)
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = re.sub('s$', '', word)
    return word


def tokenize(document):
    """Return a list whose elements are the separate words in the document.

    Args:
        document (str): text of the document.

    Returns:
        list[str]: words in the document.

    """
    return [basic_stemming(word) for word in re.split('[^a-zA-Z]', document.lower()) if word]


def build_profiles(filename):
    """Return the vocabulary extracted from the profiles and a list whose elements are the user's profiles.

    Args:
        filename (str): absolute or relative path of the file containing the users' profiles. Each line of this file
            corresponds to one user and must be in the following format: username#interest1#interest2#interestN

    Returns:
        set[str], list[Profile]

    """
    vocabulary = set()
    profiles = list()
    with open(filename) as f:
        for profile in f.readlines():
            profile_info = profile.strip().split(DELIMITER)
            username, interests = profile_info[0], profile_info[1:]
            profiles.append(Profile(username, interests))
            # noinspection PyTypeChecker
            vocabulary.update(map(basic_stemming, interests))
    return vocabulary, profiles


def build_dictionary(filename):
    """Return a dictionary mapping tokens to terms. Similar tokens are mapped to the same terms.

    Args:
        filename (str): absolute or relative path of the dictionary file. Each line of this file corresponds to similar
            words in the following format: term#similar1#similar2#similarN

    Returns:
        dict[str, str]: allowing inverse lookups. For example:

        {
            'politics': 'politics',
            'congress': politics',
            'law': 'politics',
            'books': 'books',
            'essay': 'books',
            'Shakespeare': 'books'
        }

    """
    dictionary = dict()
    with open(filename) as f:
        for entry in f.readlines():
            similar_words = map(basic_stemming, entry.strip().split(DELIMITER))
            dictionary.update({k: similar_words[0] for k in similar_words})
    return dictionary


def count_frequency(corpus, vocabulary, dictionary):
    """Return the terms frequency for each document in the given corpus. It will only include terms that are in the
     provided dictionary (key). Similar terms (extracted from the dictionary) are considered as the same term.

     Args:
         corpus (list[str]): paths of the documents.
         vocabulary (set[str]): vocabulary of terms.
         dictionary (dict[str, str]): similar term lookup. Keys doesn't need to belong to the vocabulary. Values do.

     Return:

            {
                'document1': { 'term1': frequency1, 'term2': frequency2, 'termN': frequencyN }
                'document2': { 'term1': frequency1, 'term2': frequency2, 'termN': frequencyN }
                'documentN': { 'term1': frequency1, 'term2': frequency2, 'termN': frequencyN }

            }

    """
    term_frequency = defaultdict(lambda: {k: 0 for k in vocabulary})
    document_length = dict()
    for filename in corpus:
        with open(filename) as document:
            tokens = tokenize(document.read())
            document_id = os.path.splitext(os.path.basename(filename))[0]
            document_length[document_id] = len(tokens)
            for token in tokens:
                if token in dictionary:
                    term = dictionary[token]
                    if term in vocabulary:
                        term_frequency[document_id][term] += 1
    return term_frequency, document_length


def find_recommendations(profiles, corpus, vocabulary, dictionary,
                         score_threshold=SCORE_THRESHOLD, score_only_cos=True):
    """Rank the documents in the corpus for each user according to their interests.

    Args:
        profiles (list[Profile]): list of profiles.
        corpus (list[str]): paths of the documents.
        vocabulary (set[str]): vocabulary of terms.
        dictionary (dict[str, str]): similar term lookup. Keys doesn't need to belong to the vocabulary. Values do.
        score_threshold (float): when computing the score, ignore recommendations with an score lower than this value.
        score_only_cos (bool): if True the score is the cosine between the profile and the document. If False, it is a
            product of the cosine and a ratio of the relevant words for the user.

    The recommendations are stored inside the user's profiles objects. See Profile#show_recommendations.

    """
    print(vocabulary)
    term_frequency, document_length = count_frequency(corpus, vocabulary, dictionary)
    space = VectorSpace(vocabulary)
    for profile in profiles:
        for document_id, frequencies in term_frequency.items():
            query = space.as_normalized_vector({basic_stemming(k): 1 for k in profile.interests})
            document = space.as_normalized_vector(frequencies)
            score = space.cos(query, document)
            if not score_only_cos:
                relevant_terms = sum([frequencies[basic_stemming(i)] for i in profile.interests])
                total_terms = float(document_length[document_id])
                ratio = relevant_terms / total_terms
                score *= ratio * SCORE_MULTIPLIER
            if score > score_threshold:
                profile.add_recommendation(document_id, score)
    return term_frequency


def is_tokenizable(filename):
    """Return True if the file can be tokenized, False otherwise.

    Args:
        filename (str): absolute or relative path of a file.

    Returns:
        bool: True if the file can be tokenized, False otherwise.

    """
    filename, extension = os.path.splitext(filename)
    return extension in VALID_EXTENSIONS


def files_in_dir(dir_path):
    """Return a list with all the files in the given path.

    Args:
        dir_path (str): absolute or relative path of a directory.

    Returns:
        List[str]: a list of filename.

    """
    return ['{}/{}'.format(dir_path, f) for f in os.listdir(dir_path) if is_tokenizable(f)]


def main():
    vocabulary, profiles = build_profiles(PROFILES_FILE)
    dictionary = build_dictionary(DICTIONARY_FILE)
    corpus = files_in_dir(DOCUMENTS_DIR)
    frequencies = find_recommendations(profiles, corpus, vocabulary, dictionary,
                                       score_threshold=SCORE_THRESHOLD, score_only_cos=False)
    print('*' * 42)
    print('Terms frequencies (similar grouped)'.center(42, ' '))
    print('*' * 42)
    pprint.pprint(dict(frequencies))
    print('*' * 42)
    map(lambda profile: profile.show_recommendations(), profiles)
    print('Documents with score less than {} are hidden'.format(SCORE_THRESHOLD))


if __name__ == '__main__':
    main()
