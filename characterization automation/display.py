import matplotlib.pyplot as plt
import numpy as np
from models import Stats

"""
collection of functions to display various different scenarios in coast down calculation
"""


def display_plot(x: np.ndarray, y: np.ndarray, title, show=True):
    plt.plot(x, y, 'b')
    plt.title(title)
    if show:
        plt.show()


def display_prediction(t: np.ndarray, v: np.ndarray, accel: np.ndarray, brake: np.ndarray,
                       prediction: (np.ndarray, np.ndarray), stats: Stats,
                       title, show=True):
    """
    displays predicted velocity against actual velocity
    """
    plt.plot(t, v, 'b', label='Derived Velocity [m/s]')
    plt.plot(t, accel, 'g', label='Acceleration Pressed {0, 1}')
    plt.plot(t, brake, 'r', label='Brake Pressed {0, 1}')
    plt.plot(prediction[0], prediction[1], 'm', label='Model Velocity Prediction (SciPy odesol)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    # plt.plot([], [], ' ', label=f'RMSE={stats.rmse}')

    plt.legend()
    plt.title(title)
    if show:
        plt.show()


def display_RR_model(velocities: np.ndarray, accelerations: np.ndarray, model: np.poly1d, file_name):
    lin = np.linspace(velocities.min(), velocities.max(), 100)
    plt.plot(lin, model(lin), c='g', label='Model')
    plt.scatter(velocities, accelerations, c='b')
    plt.title(f'Acceleration vs. Velocity [{file_name}]')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Acceleration [m/s/s]')
    plt.legend()
    plt.show()


def display_peaks(time: np.ndarray, velocity: np.ndarray, peaks, file_name):
    """
    Displays raw peaks as a plot
    :param file_name:
    :param time: x-axis
    :param velocity: y-axis
    :param peaks: peaks to detect
    :return:
    """
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Raw Peak Detection [{file_name}]')

    plt.plot(time, velocity, 'b', label='Derived velocity')
    plt.scatter(time[peaks], velocity[peaks], c='r')
    plt.legend()
    plt.show()


def display_intervals(time: np.ndarray, velocity: np.ndarray, intervals, file_name):
    """
    Displays all start and stop intervals on velocity plot
    :param file_name:
    :param time: x-axis
    :param velocity: y-axis
    :param intervals: intervals to display
    :return:
    """
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Coasting Intervals [{file_name}]')

    start, stop = map(list, zip(*intervals))

    plt.plot(time, velocity, 'b', label='Derived velocity')
    plt.scatter(time[start], velocity[start], c='g')
    plt.scatter(time[stop], velocity[stop], c='r')
    plt.legend()
    plt.show()
