import os

import numpy as np

import automata

BASE_PATH = os.path.dirname(__file__)

def test_sandpile():

    initial64 = np.load(os.sep.join((BASE_PATH, 'pile_64x64_init.npy')))
    final64 = np.load(os.sep.join((BASE_PATH, 'pile_64x64_final.npy')))

    assert (automata.sandpile(initial64) == final64).all()

    initial128 = np.load(os.sep.join((BASE_PATH, 'pile_128x128_init.npy')))
    final128 = np.load(os.sep.join((BASE_PATH, 'pile_128x128_final.npy')))

    assert (automata.sandpile(initial128) == final128).all()

def test_life():

    initialGlider = np.array([[False, False, False, False, False],
                              [False, True, True, True, False],
                              [False, False, False, True, False],
                              [False, False, True, False, False],
                              [False, False, False, False, False]])
    finalGlider = np.array([[[False, False, True, True, True],
                             [False, False, False, False, True],
                             [False, False, False, True, False],
                             [False, False, False, False, False],
                             [False, False, False, False, False],]])
    assert (automata.life(initialGlider, 4) == finalGlider).all()
    assert (automata.life(initialGlider, 4, periodic=True) == finalGlider).all()

    initialBlinker = np.array([[False, False, False, False, False],
                               [False, False, False, False, False],
                               [False, True, True, True, False],
                               [False, False, False, False, False],
                               [False, False, False, False, False]])
    finalBlinker = np.array([[False, False, False, False, False],
                             [False, False, True, False, False],
                             [False, False, True, False, False],
                             [False, False, True, False, False],
                             [False, False, False, False, False]])
    assert (automata.life(initialBlinker, 1) == finalBlinker).all()
    assert (automata.life(initialBlinker, 1, periodic=True) == finalBlinker).all()

def test_lifetri():

    initialTri = np.zeros((10, 10), bool)
    initialTri[4:6, 3:7] = True
    finalTri = np.zeros((10, 10), bool)
    finalTri[4:6, 5:9] = True
    assert (automata.lifetri(initialTri, 3) == finalTri).all()

def test_life_generic():
    fiveMatrix = np.array([[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1 ,0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1 ,1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1 ,1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0 ,0 ,0 ,1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0 ,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              [0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1 ,0, 1, 0, 0, 1, 1, 1, 0, 0],
              [0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 ,1, 0, 1, 0 ,0, 1, 1, 1, 0],
              [0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 1, 1, 0 ,0, 1, 0 ,1 ,0 ,0, 1, 1 ,1],
              [1 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0 ,1 ,0, 0, 1, 1],
              [1 ,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,1, 1, 0, 0, 1 ,0 ,1, 0, 0, 1],
              [1 ,1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 1, 1 ,1, 0 ,0 ,1 ,0, 1, 0, 0],
              [0 ,1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 1, 1 ,1, 0 ,0 ,1, 0, 1, 0],
              [0 ,0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1 ,1, 1 ,0 ,0, 1, 0, 1],
              [1 ,0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ,1 ,0, 0, 1, 0]])
    initialGlider = np.array([[False, False, False, False, False],
                              [False, True, True, True, False],
                              [False, False, False, True, False],
                              [False, False, True, False, False],
                              [False, False, False, False, False]]).reshape(25,)
    finalGlider = np.array([False, False, True, True, True,
                             False, False, False, False, True,
                             False, False, False, True, False,
                             False, False, False, False, False,
                             False, False, False, False, False])
    environment = {2, 3}
    fertility = {3}
    assert (automata.life_generic(fiveMatrix, initialGlider, 4, environment, fertility) == finalGlider).all()
