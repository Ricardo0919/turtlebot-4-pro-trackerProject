import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ricardosierra/Documents/TEC/7Semestre/ApplicationProject/turtlebot-4-pro-trackerProject/tracker_ws/install/tracking_person'
