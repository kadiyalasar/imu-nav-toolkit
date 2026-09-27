"""Small navigation toolkit: synthetic IMU/GPS data, Allan deviation,
magnetometer calibration, heading filters, dead reckoning, and an EKF.

Conventions used EVERYWHERE in this package (write them down, check them first):
  * World frame: ENU  (x = East, y = North, z = Up)
  * Body frame:  FLU  (x = Forward, y = Left, z = Up)   -- ROS REP-103
  * Yaw (psi):   angle from East, counter-clockwise positive, radians
  * Gyro z > 0   means the vehicle is turning LEFT (counter-clockwise)
  * Units: SI (m, s, rad). Magnetometer in gauss.
"""
