# Characterization Automation

This script is built to automatically generate coefficients for rolling resistance and aerodynamic drag for SR-3.

Using a coastdown method, the car is accelerated to a certain speed on a level surface and allowed to coast down to 0 m/s.
The only forcees that are affecting the car during this coast down interval is rolling resistance and aerodynamic drag.

From experimental measurements, it's known that we can model the coefficients of rolling resistance as a linear function of velocity
and aerodynamic drag as a quadratic function. By plotting an Acceleration vs. Velocity graph during coast down, we can use polynomial fit
to determine the coefficients of these functions.

This gives us the following equation.
https://render.githubusercontent.com/render/math?math=a = crr1 + crr2 * v + k * v^2

This script also tests how well the model is at predicting velocity from actual historical data. Using topological analysis, the program
will automatically select intervals during a race where the car is coasting. Then, we can use our model of rolling resistance to predict
velocity. This can be done by solving the differential equation shown above for velocity. Unfortuantely, it evaluates to a pretty ugly
integral, so numerical approximation using Euler approximations is used to model velocity. This allows us to see how well the model
is at predicting outside the testing data.
