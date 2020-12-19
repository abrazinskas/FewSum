import unittest
from fewsum.utils.helpers.search import find_mirror_next


class TestSearchHelpers(unittest.TestCase):

    def test_find_mirror_next(self):
        MIRROR_CENTRE = ['AND', ',']
        seq = ['0', '!', '!@#!', "a", 'b', 'c', "AND", "a", 'b']
        n = 3
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) == 'c')

        seq = ["AND", "AND", "a", 'b']
        n = 3
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) is None)

        seq = ["a", "b", "a", "AND", "a"]
        n = 3
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) is None)

        seq = ["a", "b", "a", "AND", "a"]
        n = 1
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) is None)

        seq = ["a", "a", "b"]
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) is None)

        seq = ['a', 'b', 'AND', 'c']
        self.assertTrue(find_mirror_next(seq, max_window_size=3,
                               mirror_centre=MIRROR_CENTRE) is None)

        seq = ["a", "AND"]
        self.assertTrue(find_mirror_next(seq, max_window_size=3,
                               mirror_centre=MIRROR_CENTRE) == "a")

        seq = ["a", "b", "d", "AND", "a", "b"]
        n = 3
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) == "d")

        seq = ['AND']
        n = 55
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) is None)

        seq = ['This', 'product', 'works', 'great', 'for', 'nausea', ',']
        n = 3
        self.assertTrue(find_mirror_next(seq, n, MIRROR_CENTRE) == 'nausea')


if __name__ == '__main__':
    unittest.main()
