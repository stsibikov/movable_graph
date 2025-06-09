from typing import List


def is_iter(obj) -> bool:
    '''
    check if value is an iterable, with string not being counted as an iterable
    '''
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def to_list(v) -> List:
    if v is None:
        return None
    if not is_iter(v):
        return [v]
    return list(v)
