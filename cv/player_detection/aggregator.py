import math


class DataStreamAggregator:
    def __init__(self):
        self.n = 0  # Number of data points
        self.mean = 0  # Mean of the data points
        self.M2 = (
            0  # Sum of squares of differences from the mean (used to compute variance)
        )

    def update(self, new_value):
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_stdv(self):
        if self.n < 2:
            return float("nan")  # Not enough data to compute standard deviation
        variance = self.M2 / self.n
        return math.sqrt(variance)

    def get_n(self):
        return self.n

    def is_noise(self, value):
        return abs(self.get_mean() - value) > self.get_stdv() * 2
