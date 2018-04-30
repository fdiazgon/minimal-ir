import math
import unittest

import profileir


class ProfileRetrievalTest(unittest.TestCase):

    def testStemming(self):
        self.assertEqual('caress', profileir.basic_stemming('caresses'), msg='Rule sses failed')
        self.assertEqual('poni', profileir.basic_stemming('ponies'), msg='Rule ies failed')
        self.assertEqual('caress', profileir.basic_stemming('caress'), msg='Rule ss failed')
        self.assertEqual('cat', profileir.basic_stemming('cats'), msg='Rule s failed')
        self.assertEqual('deep', profileir.basic_stemming('deep'), msg='No rule should be applied')

    def testTokenize(self):
        document = 'The output should-contain, six words?'
        tokens = profileir.tokenize(document)
        self.assertListEqual(['the', 'output', 'should', 'contain', 'six', 'word'], tokens)

    def testProfiles(self):
        terms, profiles = profileir.build_profiles(profileir.PROFILES_FILE)
        self.assertTrue(all(term in terms for term in {'movi', 'politic', 'soccer'}))
        profile1, profile2, profile3 = profiles[0], profiles[1], profiles[2]
        self.assertEquals(profile1.name, 'User1')
        self.assertSetEqual(profile1.interests, {'movies', 'politics'})
        self.assertEquals(profile2.name, 'User2')
        self.assertSetEqual(profile2.interests, {'politics', 'soccer'})
        self.assertEquals(profile3.name, 'User3')
        self.assertSetEqual(profile3.interests, {'politics'})

    def testVectorSpace(self):
        axes = {'movies', 'politics', 'soccer'}
        space = profileir.VectorSpace(axes)
        vector = space.as_normalized_vector({'movies': 2, 'politics': 2, 'soccer': 1})
        # sqrt(2**2 + 2**2 + 1**2) = 3
        self.assertItemsEqual([2 / 3., 2 / 3., 1 / 3.], vector)

        def modulus(v): return sum([c ** 2 for c in v])

        self.assertAlmostEqual(1, modulus(vector), delta=0.1, msg='Modulus of the vector is not 1')

        interests1 = {'movies': 1, 'politics': 1}
        profile1 = space.as_normalized_vector(interests1)
        self.assertAlmostEquals(1, modulus(profile1), delta=1e-10, msg='Modulus of profile1 is not 1')
        interests2 = {'politics': 1, 'soccer': 1}
        profile2 = space.as_normalized_vector(interests2)
        self.assertAlmostEquals(1, modulus(profile2), delta=1e-10, msg='Modulus of profile2 is not 1')
        interests3 = {'soccer': 1}
        profile3 = space.as_normalized_vector(interests3)
        self.assertAlmostEquals(1, modulus(profile3), delta=1e-10, msg='Modulus of profile3 is not 1')

        self.assertAlmostEqual(math.cos(math.radians(60)), space.cos(profile1, profile2), delta=1e-10)
        self.assertAlmostEqual(math.cos(math.radians(45)), space.cos(profile2, profile3), delta=1e-10)
        self.assertAlmostEqual(1, space.cos(profile1, profile1), delta=1e-10)
        self.assertAlmostEqual(0, space.cos(profile1, profile3), delta=1e-10)

        print('***** VECTOR SPACE TEST *****')
        print('Profiles in space: {}'.format(space.axes))
        print('Profile 1 ({}): -> {}'.format(interests1, profile1))
        print('Profile 2 ({}): -> {}'.format(interests2, profile2))
        print('Profile 3 ({}): -> {}'.format(interests3, profile3))
        print('*****************************')

    def testDictionary(self):
        dictionary = profileir.build_dictionary(profileir.DICTIONARY_FILE)
        self.assertDictContainsSubset({'movi': 'movi', 'classic': 'movi', 'review': 'movi'}, dictionary)
        self.assertDictContainsSubset({'politic': 'politic', 'media': 'politic', 'voter': 'politic'}, dictionary)
        self.assertDictContainsSubset({'soccer': 'soccer', 'league': 'soccer', 'victory': 'soccer'}, dictionary)

    def testCountFrequency(self):
        corpus = ['{}/blade-runner.txt'.format(profileir.DOCUMENTS_DIR)]
        vocabulary = {'movi', 'politic', 'soccer'}
        dictionary = {'blade': 'movi', 'runner': 'movi'}
        frequency, document_length = profileir.count_frequency(corpus, vocabulary, dictionary)
        self.assertEqual(4, frequency['blade-runner']['movi'])
        self.assertEqual(0, frequency['blade-runner']['politic'])
        self.assertEqual(0, frequency['blade-runner']['soccer'])
        self.assertEqual(1, len(document_length))

    def testProfileScores(self):
        profile1 = profileir.Profile('User1', ['movies'])
        profile2 = profileir.Profile('User2', ['movies', 'politics'])
        profile3 = profileir.Profile('User3', ['politics'])
        profile4 = profileir.Profile('User4', ['soccer'])
        profiles = [profile1, profile2, profile3, profile4]
        document_id = 'blade-runner'
        corpus = ['{}/{}.txt'.format(profileir.DOCUMENTS_DIR, document_id)]
        vocabulary = {'movi', 'politic', 'soccer'}
        dictionary = {'blade': 'movi', 'runner': 'movi'}
        profileir.find_recommendations(profiles, corpus, vocabulary, dictionary,
                                       score_threshold=0, score_only_cos=True)

        self.assertEquals(1, profile1.recommendations[document_id])
        self.assertEquals(math.cos(math.radians(45)), profile2.recommendations[document_id])
        self.assertEquals(0, len(profile3.recommendations))

        print('***** PROFILE SCORES TEST *****')
        map(lambda profile: profile.show_recommendations(), profiles)
        print('*******************************')

        profileir.find_recommendations(profiles, corpus, vocabulary, dictionary,
                                       score_threshold=0, score_only_cos=False)
        self.assertTrue(profile1.recommendations[document_id] < 1)

    def testIsTokenizable(self):
        self.assertTrue(profileir.is_tokenizable('file.txt'))
        self.assertFalse(profileir.is_tokenizable('file.xml'))

    def testFilesInDir(self):
        documents = profileir.files_in_dir(profileir.DOCUMENTS_DIR)
        self.assertIn('{}/{}'.format(profileir.DOCUMENTS_DIR, 'film-quiz.txt'), documents)
