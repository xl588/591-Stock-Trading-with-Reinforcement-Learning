import numpy as np
from numpy.random import binomial
import itertools

def sigmoid_function(alpha):
    return lambda x: 1. /(1 + (1 /np.where(x == 0, 1e-8, x) -1) ** alpha)

class Process:

    def __init__(self, alpha):
        self.alpha = alpha

    class State:

        def __init__(self, num_up_moves, num_down_moves):
            self.num_up_moves = num_up_moves
            self.num_down_moves = num_down_moves

    def up_prob(self, state):
        total = state.num_up_moves + state.num_down_moves
        return sigmoid_function(self.alpha)(
            state.num_down_moves / total
        ) if total else 0.5

    def next_state(self, state):
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process.State(
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move
        )


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


def process_price_traces(
        start_price,
        alpha,
        time_steps,
        num_traces):
    process = Process(alpha)
    start_state = Process.State(num_up_moves=0, num_down_moves=0)
    return np.vstack([
        np.fromiter((start_price + s.num_up_moves - s.num_down_moves
                     for s in itertools.islice(simulation(process, start_state),
                                               time_steps + 1)), float)
        for _ in range(num_traces)])

