'''
Testing functions for the first neusci440 homework

@ Jeremy Schroeter, 2024
'''

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from math import isclose


def test_s1_p1(func: callable) -> None:
    '''
    Test pythagorean formula function
    '''
    first_side = [3, 2, 5]
    second_side = [4, 3, 3]
    third_side = [5.0, 3.605551275463989, 5.830951894845301]

    for (a, b, c) in list(zip(first_side, second_side, third_side)):
        assert isclose(func(a, b), c)
    print('All tests passed')


def test_s1_p2(func: callable) -> None:
    '''
    Test average function
    '''
    test_lists = [np.random.random(5) for _ in range(3)]
    for test in test_lists:
        assert isclose(func(test), np.mean(test))
    print('All tests passed')


def test_s1_p3(func: callable) -> None:
    '''
    Test variance function
    '''
    test_lists = [np.random.random(5) for _ in range(3)]
    for test in test_lists:
        assert isclose(func(test), np.var(test))
    print('All tests passed')


def test_s1_p4(func: callable) -> None:
    '''
    Test max function
    '''
    test_lists = [np.random.random(5) for _ in range(3)]
    for test in test_lists:
        assert isclose(func(test), max(test))
    print('All tests passed')


def test_s2_p1(func: callable) -> None:
    '''
    Test arange function
    '''
    ab = [(5, 8), (0, 45), (-10, 70)]
    for (a, b) in ab:
        assert_array_equal(func(a, b), np.arange(a, b))
    print('All tests passed')


def test_s2_p2(func: callable) -> None:
    '''
    Test slicing function
    '''
    test_arrays = [
        np.random.random(8 * 3 * 46).reshape(8, 3, 46),
        np.random.random(12 * 3 * 4).reshape(12, 3, 4),
        np.random.random(7 * 7 * 7).reshape(7, 7, 7)
    ]

    for test in test_arrays:
        assert_array_almost_equal(func(test), test[:3, :, -2:])
    print('All tests passed')


def test_s2_p3_1(func: callable) -> None:
    '''
    Test first gaussian function
    '''

    for i in range(3):
        test = np.random.random(50)
        assert_array_almost_equal(func(test), np.exp(-test**2))
    print('All tests passed')


def test_s2_p3_2(func: callable) -> None:
    '''
    Test second gaussian function
    '''
    aa = [-5, 1.2, 0.59]
    bb = [100, -1, 0.001]
    cc = [4, -10, np.pi]

    for (a, b, c) in list(zip(aa, bb, cc)):
        test = np.random.random(50)
        assert_array_almost_equal(func(test, a, b, c), a * np.exp(-(test - b)**2) + c)
    print('All tests passed')


def test_s2_p4_1(func: callable) -> None:
    '''
    Test averaging function
    '''
    arr = np.random.random(10 * 5 * 10000).reshape(10, 5, 10000)
    assert_array_almost_equal(func(arr), arr.mean(1))
    print('All tests passed')


def test_s2_p4_2(func: callable) -> None:
    '''
    Test averaging function
    '''
    arr = np.random.random(10 * 5 * 10000).reshape(10, 5, 10000)
    assert_array_almost_equal(func(arr), arr.std(1))
    print('All tests passed')


def test_s2_p6(func: callable) -> None:
    '''
    Test array filling function
    '''
    a, b = 10, 15
    arr = func(a, b)
    for row in range(a):
        assert arr[row].sum() > 0
    print('All tests passed')
