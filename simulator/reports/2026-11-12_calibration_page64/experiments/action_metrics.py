"""Compare target IDs independently of integer versus integer-string encoding."""
import json
import re

METRIC_VERSION = 'integer_target_ids_v2'


def canonical_index(value):
    """Normalize unambiguous integer IDs without merging booleans or floats."""
    if type(value) is int:
        return ('integer', value)
    if isinstance(value, str) and re.fullmatch(r'[+-]?[0-9]+', value.strip()):
        return ('integer', int(value.strip()))
    return (type(value).__name__, json.dumps(value, ensure_ascii=False, sort_keys=True))


def target_indices_equal(left, right):
    """Preserve target order and multiplicity, and require extracted lists."""
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    return [canonical_index(x) for x in left] == [canonical_index(x) for x in right]
