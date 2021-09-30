from collections import namedtuple

Telemetry = namedtuple('Telemetry', ['velocity', 'time', 'df', 'file'])
Configuration = namedtuple('Config', ['train', 'test', 'mass', 'wheel_radius', 'peak_prominence', 'debug',
                                      'min_interval_size', 'max_accel_held', 'max_brake_held', 'n_poly_degree',
                                      'prune', 'block_width', 'block_height_threshold', 'resolution'])
Stats = namedtuple('Stats', ['rmse'])
