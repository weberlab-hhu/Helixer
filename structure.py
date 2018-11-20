import json
import copy


class GenericData(object):
    """Handles basic to/from json conversions for self and any sub data it holds"""
    def __init__(self):
        # attribute name, exported_to_json, expected_inner_type, data_structure
        # where data structure is always `None` and type the result of `type(self.attr)` unless `expected_inner_type`
        # is an instance of this class (GenericData), then `data_structure` denotes any grouping class (e.g. list),
        # or is still `None` if not grouped at all
        self.spec = [('spec', False, list, None), ]

    def __str__(self):
        return str(self.__dict__)

    def to_json(self, json_path):
        jsonable = self.to_jsonable()
        with open(json_path, 'w') as f:
            json.dump(jsonable, f, indent=2, separators=(',', ': '))

    def from_json(self, json_path):
        with open(json_path) as f:
            jsonable = json.load(f)
        self.load_jsonable(jsonable)

    def to_jsonable(self):
        out = {}
        for key in copy.deepcopy(list(self.__dict__.keys())):
            raw = self.__getattribute__(key)
            cleaned, is_exported = self._prep_main(key, raw, towards_json=True)
            if is_exported:
                out[key] = cleaned
        return out

    def load_jsonable(self, jsonable):
        for key in jsonable:
            raw = jsonable[key]
            cleaned, is_exported = self._prep_main(key, raw, towards_json=False)
            assert is_exported, "Expected only exported attributes to be found in the json, error at {}".format(
                key
            )
            self.__setattr__(key, cleaned)

    def _prep_main(self, key, raw, towards_json=True):
        _, is_exported, expected_type, data_structure = self.get_key_spec(key)
        if is_exported:
            if data_structure is None:
                out = self._prep_none(expected_type, raw, towards_json)
            elif data_structure in (list, tuple):  # maybe could also have set and iter, but idk why you'd need this
                out = self._prep_list_like(expected_type, raw, data_structure, towards_json)
            elif data_structure is dict:
                out = self._prep_dict(expected_type, raw, towards_json)
            else:
                raise ValueError("no export method prepared for data_structure of type: {}".format(data_structure))
        else:
            out = None
        return copy.deepcopy(out), is_exported

    def get_key_spec(self, key):
        key_spec = [s for s in self.spec if s[0] == key]
        assert len(key_spec) == 1, "{} attribute has {} instead of 1 matches in spec".format(key, len(key_spec))
        return key_spec[0]

    @staticmethod
    def _confirm_type(expected_type, to_check):
        # allow either None or the expected_type
        if to_check is not None:
            assert isinstance(to_check, expected_type), \
                "type: ({}) differs from expectation in spec ({}), for ({})".format(
                type(to_check), expected_type, to_check
            )

    def _prep_none(self, expected_type, raw, towards_json=True):
        if towards_json:
            return self._prep_none_to_json(expected_type, raw)
        else:
            return self._prep_none_from_json(expected_type, raw)

    def _prep_none_to_json(self, expected_type, raw):
        out = raw
        self._confirm_type(expected_type, out)
        if issubclass(expected_type, GenericData):
            out = out.to_jsonable()
        return out

    def _prep_none_from_json(self, expected_type, raw):
        out = raw
        if issubclass(expected_type, GenericData):
            out = expected_type()
            out.load_jsonable(raw)
        self._confirm_type(expected_type, out)
        return out

    def _prep_list_like(self, expected_type, raw, data_structure, towards_json=True):
        out = []
        for item in raw:
            out.append(self._prep_none(expected_type, item, towards_json))
        if not towards_json:
            out = data_structure(out)
        return out

    def _prep_dict(self, expected_type, raw, towards_json=True):
        out = {}
        for key in raw:
            out[key] = self._prep_none(expected_type, raw[key], towards_json)
        return out


def add_paired_dictionaries(add_to, add_from):
    add_to = copy.deepcopy(add_to)
    for key in add_from:
        if key not in add_to:
            add_to[key] = copy.deepcopy(add_from[key])
        elif isinstance(add_to[key], dict):
            add_to[key] = add_paired_dictionaries(add_to[key], add_from[key])
        else:
            add_to[key] += add_from[key]
    return add_to
