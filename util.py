import numpy as np
from math import comb, ceil
from scipy.signal import find_peaks
from display import display_peaks, display_intervals, display_RR_model, display_interval_plot
import pandas as pd
import yaml
import scipy.ndimage
from models import Stats, Telemetry, Configuration


def load_config(config='config.yaml'):
    """
    Load data from configuration file
    :param config:
    :return:
    """
    with open(config) as stream:
        data = yaml.safe_load(stream)
        return Configuration(**data)


def process_data(config: Configuration, source_file) -> Telemetry:
    """
    Process telemetry data and derive columns
    :param config:
    :param source_file: source file to parse data from
    :return:
    """
    telemetry = pd.read_csv(source_file)
    time = np.array((telemetry['time'] - telemetry['time'][0]) / 1000)

    rpm = telemetry['Right_Wavesculptor_RPM']
    velocity = np.array((rpm * 2 * np.pi * config.wheel_radius / 60))

    return Telemetry(time=time, velocity=velocity, df=telemetry, file=source_file)


def secant_derivative(x: np.ndarray, y: np.ndarray):
    """
    Numerically computes derivative by taking secant slopes around a point
    :param x: x values
    :param y: y values
    :return: dy/dx
    """
    dy_dx = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dy_dx = np.append(dy_dx, dy_dx[-1])
    dy_dx = np.append(dy_dx, dy_dx[-1])
    return dy_dx


def shift_polynomial(poly, x_shift):
    """
    Computes right shifted polynomial function by expanding shift with binomial expansion

    :param poly: coefficients of nth degree polynomial
    :param x_shift: scalar shift to the right
    :return:
    """
    n = len(poly)

    n_poly = np.zeros(len(poly))
    for p in range(n):
        # calculate combinations: (x + y)^3 = [1 3 3 1]
        combs = [comb(p, i) * (1 if i % 2 == 0 else -1) for i in range(p + 1)]
        combs = np.pad(combs, (n - p - 1, 0), 'constant')
        # calculate powers from shift: (x + 2)^3 = [1 2 4 8]
        powers = [x_shift ** i for i in range(p + 1)]
        powers = np.pad(powers, (n - p - 1, 0), 'constant')
        # multiply it out and add to the polynomial
        component = poly[n - p - 1] * combs * powers
        n_poly += component
    return list(n_poly)


def generate_coasting_intervals(config: Configuration, telemetry: Telemetry):
    """
    Generates intervals where coasting occurs. Used for training and testing a model.
    :param min_interval_size: smallest size of an interval allowed
    :param df: rest of dataframe
    :param time: time values
    :param velocity: velocity values
    :param peak_prominence: prominence threshold for detecting a peak (Topological Prominence Wikipedia)
    :param debug: show graphs for debugging
    :return:
    """
    velocity = np.array(telemetry.velocity)
    accel_pressed = np.array(telemetry.df['MC2_Accel_Pressed'])
    brake_pressed = np.array(telemetry.df['MC2_Brake_Pressed'])
    velocity = velocity.clip(min=0)
    peaks = find_peaks(velocity, prominence=config.peak_prominence)[0]

    if config.debug:
        display_peaks(telemetry.time, velocity, peaks, telemetry.file)

    intervals = []
    for peak_idx in range(len(peaks)):
        def calculate_first_limit_break(arr: np.ndarray, limit):
            regions = list(scipy.ndimage.find_objects(scipy.ndimage.label(arr)[0]))
            sums = np.array([telemetry.time[r[0].stop] - telemetry.time[r[0].start] for r in regions])
            exceeds_limits_idxs = np.argwhere(sums > limit).reshape(-1)

            if len(exceeds_limits_idxs) > 0:
                first_limit_break = regions[exceeds_limits_idxs[0]][0].start
            else:
                first_limit_break = float('inf')
            return first_limit_break

        # find index of minimum in the range between this peak and the next
        if peak_idx == len(peaks) - 1:
            min_idx = np.argmin(velocity[peaks[peak_idx]:])
            first_accel = calculate_first_limit_break(accel_pressed[peaks[peak_idx]:], config.max_accel_held)
            first_brake = calculate_first_limit_break(brake_pressed[peaks[peak_idx]:], config.max_brake_held)

        else:
            min_idx = np.argmin(velocity[peaks[peak_idx]: peaks[peak_idx + 1]])
            first_accel = calculate_first_limit_break(accel_pressed[peaks[peak_idx]:peaks[peak_idx + 1]],
                                                      config.max_accel_held)
            first_brake = calculate_first_limit_break(brake_pressed[peaks[peak_idx]:peaks[peak_idx + 1]],
                                                      config.max_brake_held)
        stop_idx = peaks[peak_idx] + min(min_idx, first_accel, first_brake)
        # check if interval size meets minimum
        if telemetry.time[stop_idx] - telemetry.time[peaks[peak_idx]] > config.min_interval_size:
            intervals.append((peaks[peak_idx], stop_idx))

    if config.debug:
        display_intervals(telemetry.time, velocity, intervals, telemetry.file)

    # check if intervals are sane
    for (a, b) in intervals:
        assert b - a >= 0, "intervals weren't calculated properly"

    if not config.prune:
        return intervals
    # prune intervals that have swings
    pruned_intervals = []
    block_width = ceil(config.block_width / config.resolution)
    for interval in intervals:
        check_vels = velocity[interval[0]:min(interval[1] + 1, len(velocity))]
        # we want to create an array that can be divided into 'check_width' bins
        # pad the end with zeros
        check_vels = np.pad(check_vels, (0, block_width - (len(check_vels) % block_width)), 'constant')
        # our goal now is to take the mean of every 'check_width' values
        check_vels = check_vels.reshape(-1, block_width)
        check_vels = np.mean(check_vels, axis=1)
        # find the difference between the previous value
        check_vels = np.diff(check_vels)
        # check if in an interval of 1 second,
        # there is a jump in the average velocity of more than 2mph
        if np.max(check_vels) <= config.block_height_threshold:
            pruned_intervals.append(interval)
    return pruned_intervals


def generate_model_fit(config: Configuration):
    """
    Fits acceleration to velocity curve
    :param config:
    :return:
    """
    data = process_data(config, config.train)

    intervals = generate_coasting_intervals(config, data)
    velocities = np.array([])
    accelerations = np.array([])
    for a, b in intervals:
        t = data.time[a:b]
        v = data.velocity[a:b]
        a = secant_derivative(t, v)
        velocities = np.concatenate((velocities, v))
        accelerations = np.concatenate((accelerations, a))

    model = np.poly1d(np.polyfit(velocities, accelerations, config.n_poly_degree))
    if config.debug:
        display_RR_model(velocities, accelerations, model, config.train)

    return model


def euler_approximation(v0, t0, model, steps, config: Configuration):
    t, v = [t0], [v0]
    dt = config.resolution
    for i in range(steps):
        v.append(v[-1] + model(v[-1]) * dt)
        t.append(t[-1] + dt)

    return np.array(t), np.array(v)


def ivp_approx(v0, t0, model, steps, config: Configuration):
    from scipy.integrate._ivp import solve_ivp

    dt = config.resolution
    solution = solve_ivp(lambda t, v: model(v), [t0, t0 + steps * dt], [v0], max_step=0.1)
    return np.array(solution.t), np.array(solution.y).reshape(-1).clip(min=0)


def calculate_statistics(prediction, real):
    if len(prediction) == len(real):
        rmse = (prediction - real) ** 2
        rmse = np.sqrt(np.sum(rmse) / len(prediction))
        return Stats(rmse=rmse)
    else:
        assert 'prediction and real were not the same size'


def test_model_on_validation(model: np.poly1d, config: Configuration):
    telemetry = process_data(config, config.test)
    intervals = generate_coasting_intervals(config, telemetry)

    rmse_avg = 0
    for i in range(len(intervals)):
        a, b = intervals[i]
        t = telemetry.time[a:b]
        v = telemetry.velocity[a:b]
        hamast, hamasv = euler_approximation(v[0], t[0], model, (b - a - 1), config)
        pt, pv = ivp_approx(v[0], t[0], model, (b - a - 1), config)
        # pt, pv = pt[:len(t)], pv[:len(v)]
        # stats = calculate_statistics(pv, v)
        display_interval_plot(t, v,
                              np.array(telemetry.df['MC2_Accel_Pressed'])[a:b] + min(v.min(), pv.min()),
                              np.array(telemetry.df['MC2_Brake_Pressed'])[a:b] + min(v.min(), pv.min()),
                              (pt, pv), None, f'Interval #{i + 1} Plot [{config.test}]', show=False)
        import matplotlib.pyplot as plt
        plt.plot(hamast, hamasv, 'k', label="Evil Hamas Plot (Euler)")
        plt.show()
        # rmse_avg += stats.rmse
    # print('RMSE', rmse_avg / len(intervals))
