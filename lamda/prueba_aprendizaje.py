#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'juliowaissman'

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
