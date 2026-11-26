"""Clock-injected backlog-age policy, independent of networking and ML."""
import time


class BacklogPolicy:
    def __init__(self, max_lag=0.6, recovery_age=0.15, clock=time.monotonic):
        self.max_lag = float(max_lag)
        self.recovery_age = float(recovery_age)
        self.clock = clock

    def samples_to_drop(self, oldest_arrival, queued_samples, samples_per_second):
        age = max(0.0, self.clock() - oldest_arrival)
        if age <= self.max_lag:
            return 0
        keep_samples = max(0, int(self.recovery_age * samples_per_second))
        return max(0, queued_samples - keep_samples)
