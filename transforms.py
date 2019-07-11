import torch

from running_stat import RunningStat


class Transform:
    '''
    Composes several transformation and applies them sequentially

    Attributes
    ----------
    filters : list
        a list of callables

    Methods
    -------
    __call__(x)
        sequentially apply the callables in filters to the input and return the
        result
    '''

    def __init__(self, *filters):
        '''
        Parameters
        ----------
        filters : variatic argument list
            the sequence of transforms to be applied to the input of
            __call__
        '''

        self.filters = list(filters)

    def __call__(self, x):
        for f in self.filters:
            x = f(x)

        return x


class ZFilter:
    '''
    A z-scoring filter

    Attributes
    ----------
    running_stat : RunningStat
        an object that keeps track of an estimate of the mean and standard
        deviation of the observations seen so far

    Methods
    -------
    __call__(x)
        Update running_stat with x and return the result of z-scoring x
    '''

    def __init__(self):
        self.running_stat = RunningStat()

    def __call__(self, x):
        self.running_stat.update(x)
        x = (x - self.running_stat.mean) / (self.running_stat.std + 1e-8)

        return x


class Bound:
    '''
    Implements a bounding function

    Attributes
    ----------
    low : int
        the lower bound

    high : int
        the upper bound

    Methods
    -------
    __call__(x)
        applies the specified bounds to x and returns the result
    '''

    def __init__(self, low, high):
        '''
        Parameters
        ----------
        low : int
            the lower bound

        high : int
            the upper bound
        '''
        
        self.low = low
        self.high = high

    def __call__(self, x):
        x = torch.clamp(x, self.low, self.high)

        return x
