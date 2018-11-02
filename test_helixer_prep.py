import helixer_prep


def test_add_to_empty_dictionary():
    d1 = {'a': 1}
    d2 = {}
    d1_2 = helixer_prep.add_paired_dictionaries(d1, d2)
    d2_1 = helixer_prep.add_paired_dictionaries(d2, d1)
    assert d1 == d1_2
    assert d1 == d2_1


def test_add_nested_dictionaries():
    d1 = {'a': {'b': 1,
                'a': {'b': 10}},
          'b': 100}
    d2 = {'a': {'b': 1,
                'a': {'b': 20}},
          'b': 300}
    dsum = {'a': {'b': 2,
                  'a': {'b': 30}},
            'b': 400}
    d1_2 = helixer_prep.add_paired_dictionaries(d1, d2)
    d2_1 = helixer_prep.add_paired_dictionaries(d2, d1)
    print('d1_2', d1_2)
    print('d2_1', d2_1)
    print('dsum', dsum)
    assert dsum == d1_2
    assert dsum == d2_1
