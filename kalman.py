from logging import DEBUG, basicConfig, getLogger
from random import randint, random

import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(
        self,
        initial_estimate: float = random(),
        initial_est_error: float = random(),
        initial_measure_error: float = random(),
        sensor_values: list = [],
        delta: float = 0.01,
        delta_hat: float = 0.0,
        logging: bool = False,
        plotting: bool = False,
    ):
        self.estimate = initial_estimate
        self.gain = random()
        self.est_error = initial_est_error
        self.measure_error = initial_measure_error
        self.sensor_values = [70 + randint(-20, 20) for _ in range(sensor_values)]
        self.logging = logging
        self.plotting = plotting

        if self.logging:
            self.logger = getLogger(__name__)
            basicConfig(level=DEBUG, format="%(message)s")

    def calculate_kalman_gain(self) -> None:
        """calculates Kalman gain given error values"""
        self.gain = self.est_error / (self.est_error + self.measure_error)

    def update_estimate(self, sensor_value: int = 0.0) -> None:
        """updates estimate based on Kalman gain"""
        new_estimate = self.estimate + self.gain * (sensor_value - self.estimate)
        self.delta_hat = abs(new_estimate - self.estimate)
        self.estimate = new_estimate

    def calculate_estimate_error(self) -> None:
        """calculates error of the updated estimate"""
        self.est_error = (1 - self.gain) * self.est_error

    def iterative_updates(self) -> None:
        e = []
        for sensor_value in self.sensor_values:
            self.calculate_kalman_gain()
            self.update_estimate(sensor_value=sensor_value)
            self.calculate_estimate_error()
            self.logger.info(f"estimate: {self.estimate}")
            e.append(self.estimate)
        if self.plotting:
            fig = plt.figure()
            plt.plot(
                range(len(self.sensor_values)),
                self.sensor_values,
                "x",
                color="gray",
                label="sensor values",
            )
            plt.plot(
                range(len(self.sensor_values)), e, "-k", color="blue", label="Kalman estimate"
            )
            plt.legend(loc="upper left")
            plt.ylim(30, 120)
            plt.show()


if __name__ == "__main__":
    kf = KalmanFilter(
        initial_estimate=68.0,
        initial_est_error=2.0,
        initial_measure_error=4.0,
        sensor_values=500,
        logging=True,
        plotting=True,
    )
    kf.iterative_updates()

