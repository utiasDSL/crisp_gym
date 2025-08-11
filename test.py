from crisp_py.sensors.sensor import ForceTorqueSensor
from crisp_py.sensors.sensor_config import make_sensor_config

sensor = ForceTorqueSensor(
    make_sensor_config("force_torque"),
)
