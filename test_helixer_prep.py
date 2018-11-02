import helixer_prep


# testing: add_paired_dictionaries
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


# testing: class GenericData
class GDataTesting(helixer_prep.GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('expect', False, dict, None)]
        self.expect = {}  # this is what we expect the jsonable to be, assuming we don't change the attributes


class SimpleGData(GDataTesting):
    def __init__(self):
        super().__init__()
        # attribute name, exported_to_json, expected_inner_type, data_structure
        self.spec += [('some_ints', True, int, list),
                      ('a_string', True, str, None)]
        self.some_ints = [1, 2, 3]
        self.a_string = 'abc'
        self.expect = {'some_ints': [1, 2, 3], 'a_string': 'abc'}


class HoldsGdata(GDataTesting):
    def __init__(self):
        super().__init__()
        self.spec += [('a_gdata', True, SimpleGData, None),
                      ('list_gdata', True, SimpleGData, list),
                      ('dict_gdata', True, SimpleGData, dict)]
        self.a_gdata = SimpleGData()
        self.list_gdata = [SimpleGData()]
        self.dict_gdata = {'x': SimpleGData()}
        sgd_expect = self.a_gdata.expect
        self.expect = {'a_gdata': sgd_expect,
                       'list_gdata': [sgd_expect],
                       'dict_gdata': {'x': sgd_expect}}


def test_to_json_4_simplest_of_data():
    x = SimpleGData()
    assert x.to_jsonable() == x.expect


def test_to_json_4_recursive_generic_datas():
    x = HoldsGdata()
    print(x.to_jsonable())
    assert x.to_jsonable() == x.expect


def test_from_json_gdata():
    # make sure we get the same after export, as export->import->export
    x = SimpleGData()
    xjson = x.to_jsonable()
    xjson['a_string'] = 'new_string'
    y = SimpleGData()
    y.load_jsonable(xjson)
    assert y.to_jsonable() == xjson
    # check as above but for more complicated data holder
    holds = HoldsGdata()
    holds.a_gdata = y
    holdsjson = holds.to_jsonable()
    assert holdsjson["a_gdata"]["a_string"] == 'new_string'
    print(holdsjson)
    yholds = HoldsGdata()
    yholds.load_jsonable(holdsjson)
    assert yholds.to_jsonable() == holdsjson
