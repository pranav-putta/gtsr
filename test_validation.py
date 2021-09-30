import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import calculate_coasting_intervals, display_intervals

model = np.poly1d([-0.00042652, 0.00724936, -0.03306493, -0.01832687, -0.02292682])

car_mass = 360  # kg
wheel_radius = 0.278  # meters

# index telemetry by second
telemetry = pd.read_csv('validation.csv')
telemetry['time'] = (telemetry['time'] - telemetry['time'][0]) / 1000
telemetry.set_index('time', inplace=True)

time = telemetry.index

rpm = telemetry['Right_Wavesculptor_RPM']
velocity_mph = np.array((rpm * 2 * np.pi * wheel_radius * 60) * 0.000621371)
velocity_ms = np.array((rpm * 2 * np.pi * wheel_radius / 60))


def euler_approximation(v0, a, steps):
    t, v = [0], [v0]
    dt = 0.25
    for i in range(steps):
        v.append(v[-1] + a(v[-1]) * dt)
        t.append(t[-1] + dt)
    plt.plot(t, v, 'g', label='Model Approximation of Velocity')


intervals = calculate_coasting_intervals(velocity_mph, jump_threshold=0.5)

display_intervals(intervals, velocity_mph)
plt.show()

for a, b in intervals:
    plt.plot(np.arange(b - a), velocity_ms[a:b], 'b', label='Real Velocity')
    euler_approximation(velocity_ms[a], model, (b - a) * 4)
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.show()
