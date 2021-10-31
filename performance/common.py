COLOR = [
    (1, 0, 0),  # red
    (0, 1, 0),  # green
    (0, 0, 1),  # blue
    (1, 1, 0),  # yellow
    (0, 1, 1),  # cyan
    (1, 0, 1),  # magenta
    (0, 0, 0),  # black
    (1, 1, 1)  # white
]

TRACKER_COLORS = {
    'AdaptiveKalmanSiam': (1, 0, 0),  # red
    'KalmanSiam': (0, 1, 0),  # green
    'SiamFC': (0, 0, 1),  # blue
    'AdaptiveKalmanSiam-Single': (0, 0, 0),  # black
    'AdaptiveKalmanSiam-Distinct': (0, 1, 1),  # cyan
    'AdaptiveKalmanSiam-P': (1, 0, 1),  # magenta
}

LINE = [
    '-',  # solid line style
    '--',  # dashed line style
    '-.',  # dash-dot line style
    ':',  # dotted line style
    'solid',  # solid marker
    'dashed',  # dashed marker
    'dashdot',  # dashdot marker
    'dotted',  # dotted marker
    'None',  # none marker
]

TRACKER_LINES = {
    'AdaptiveKalmanSiam': '-',  # solid line style
    'KalmanSiam': '--',  # dashed line style
    'SiamFC': '-.'  # dash-dot line style
}

GROUNDTRUTH_EXTENSIONS = [
    'txt'
]

IMAGE_EXTENSIONS = [
    'bmp',
    'jpeg',
    'jpg',
    'tiff',
    'tif',
    'png'
]

DATASETS = [
    'OTB100',
    'TC128'
]

TRACKERS = [
    'SiamFC',
    'KalmanSiam',
    'AdaptiveKalmanSiam'
]

OTB100_SEQUENCES_ATTRIBUTES = [
    ('Basketball', ['IV', 'OCC', 'DEF', 'OPR', 'BC']),
    ('Biker', ['SV', 'OCC', 'MB', 'FM', 'OPR', 'OV', 'LR']),
    ('Bird1', ['DEF', 'FM', 'OV']),
    ('BlurBody', ['SV', 'DEF', 'MB', 'FM', 'IPR']),
    ('BlurCar2', ['SV', 'MB', 'FM']),
    ('BlurFace', ['MB', 'FM', 'IPR']),
    ('BlurOwl', ['SV', 'MB', 'FM', 'IPR']),
    ('Bolt', ['OCC', 'DEF', 'IPR', 'OPR']),
    ('Box', ['IV', 'SV', 'OCC', 'MB', 'IPR', 'OPR', 'OV', 'BC', 'LR']),
    ('Car1', ['IV', 'SV', 'MB', 'FM', 'BC', 'LR']),
    ('Car4', ['IV', 'SV']),
    ('CarDark', ['IV', 'BC']),
    ('CarScale', ['SV', 'OCC', 'FM', 'IPR', 'OPR']),
    ('ClifBar', ['SV', 'OCC', 'MB', 'FM', 'IPR', 'OV', 'BC']),
    ('Couple', ['SV', 'DEF', 'FM', 'OPR', 'BC']),
    ('Crowds', ['IV', 'DEF', 'BC']),
    ('David', ['IV', 'SV', 'OCC', 'DEF', 'MB', 'IPR', 'OPR']),
    ('Deer', ['MB', 'FM', 'IPR', 'BC', 'LR']),
    ('Diving', ['SV', 'DEF', 'IPR']),
    ('DragonBaby', ['SV', 'OCC', 'MB', 'FM', 'IPR', 'OPR', 'OV']),
    ('Dudek', ['SV', 'OCC', 'DEF', 'FM', 'IPR', 'OPR', 'OV', 'BC']),
    ('Football', ['OCC', 'IPR', 'OPR', 'BC']),
    ('Freeman4', ['SV', 'OCC', 'IPR', 'OPR']),
    ('Girl', ['SV', 'OCC', 'IPR', 'OPR']),
    ('Human3', ['SV', 'OCC', 'DEF', 'OPR', 'BC']),
    ('Human4', ['IV', 'SV', 'OCC', 'DEF']),
    ('Human6', ['SV', 'OCC', 'DEF', 'FM', 'OPR', 'OV']),
    ('Human9', ['IV', 'SV', 'DEF', 'MB', 'FM']),
    ('Ironman', ['IV', 'SV', 'OCC', 'MB', 'FM', 'IPR', 'OPR', 'OV', 'BC', 'LR']),
    ('Jump', ['SV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR']),
    ('Jumping', ['MB', 'FM']),
    ('Liquor', ['IV', 'SV', 'OCC', 'MB', 'FM', 'OPR', 'OV', 'BC']),
    ('Matrix', ['IV', 'SV', 'OCC', 'FM', 'IPR', 'OPR', 'BC']),
    ('MotorRolling', ['IV', 'SV', 'MB', 'FM', 'IPR', 'BC', 'LR']),
    ('Panda', ['SV', 'OCC', 'DEF', 'IPR', 'OPR', 'OV', 'LR']),
    ('RedTeam', ['SV', 'OCC', 'IPR', 'OPR', 'LR']),
    ('Shaking', ['IV', 'SV', 'IPR', 'OPR', 'BC']),
    ('Singer2', ['IV', 'DEF', 'IPR', 'OPR', 'BC']),
    ('Skating1', ['IV', 'SV', 'OCC', 'DEF', 'OPR', 'BC']),
    ('Skating2_1', ['SV', 'OCC', 'DEF', 'FM', 'OPR']),
    ('Skating2_2', ['SV', 'OCC', 'DEF', 'FM', 'OPR']),
    ('Skiing', ['IV', 'SV', 'DEF', 'IPR', 'OPR']),
    ('Soccer', ['IV', 'SV', 'OCC', 'MB', 'FM', 'IPR', 'OPR', 'BC']),
    ('Surfer', ['SV', 'FM', 'IPR', 'OPR', 'LR']),
    ('Sylvester', ['IV', 'IPR', 'OPR']),
    ('Tiger2', ['IV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR', 'OV']),
    ('Trellis', ['IV', 'SV', 'IPR', 'OPR', 'BC']),
    ('Walking', ['SV', 'OCC', 'DEF']),
    ('Walking2', ['SV', 'OCC', 'LR']),
    ('Woman', ['IV', 'SV', 'OCC', 'DEF', 'MB', 'FM', 'OPR']),
    ('Bird2', ['OCC', 'DEF', 'FM', 'IPR', 'OPR']),
    ('BlurCar1', ['MB', 'FM']),
    ('BlurCar3', ['MB', 'FM']),
    ('BlurCar4', ['MB', 'FM']),
    ('Board', ['SV', 'MB', 'FM', 'OPR', 'OV', 'BC']),
    ('Bolt2', ['DEF', 'BC']),
    ('Boy', ['SV', 'MB', 'FM', 'IPR', 'OPR']),
    ('Car2', ['IV', 'SV', 'MB', 'FM', 'BC']),
    ('Car24', ['IV', 'SV', 'BC']),
    ('Coke', ['IV', 'OCC', 'FM', 'IPR', 'OPR', 'BC']),
    ('Coupon', ['OCC', 'BC']),
    ('Crossing', ['SV', 'DEF', 'FM', 'OPR', 'BC']),
    ('Dancer', ['SV', 'DEF', 'IPR', 'OPR']),
    ('Dancer2', ['DEF']),
    ('David2', ['IPR', 'OPR']),
    ('David3', ['OCC', 'DEF', 'OPR', 'BC']),
    ('Dog', ['SV', 'DEF', 'OPR']),
    ('Dog1', ['SV', 'IPR', 'OPR']),
    ('Doll', ['IV', 'SV', 'OCC', 'IPR', 'OPR']),
    ('FaceOcc1', ['OCC']),
    ('FaceOcc2', ['IV', 'OCC', 'IPR', 'OPR']),
    ('Fish', ['IV']),
    ('FleetFace', ['SV', 'DEF', 'MB', 'FM', 'IPR', 'OPR']),
    ('Football1', ['IPR', 'OPR', 'BC']),
    ('Freeman1', ['SV', 'IPR', 'OPR']),
    ('Freeman3', ['SV', 'IPR', 'OPR']),
    ('Girl2', ['SV', 'OCC', 'DEF', 'MB', 'OPR']),
    ('Gym', ['SV', 'DEF', 'IPR', 'OPR']),
    ('Human2', ['IV', 'SV', 'MB', 'OPR']),
    ('Human5', ['SV', 'OCC', 'DEF']),
    ('Human7', ['IV', 'SV', 'OCC', 'DEF', 'MB', 'FM']),
    ('Human8', ['IV', 'SV', 'DEF']),
    ('Jogging_1', ['OCC', 'DEF', 'OPR']),
    ('Jogging_2', ['OCC', 'DEF', 'OPR']),
    ('KiteSurf', ['IV', 'OCC', 'IPR', 'OPR']),
    ('Lemming', ['IV', 'SV', 'OCC', 'FM', 'OPR', 'OV']),
    ('Man', ['IV']),
    ('Mhyang', ['IV', 'DEF', 'OPR', 'BC']),
    ('MountainBike', ['IPR', 'OPR', 'BC']),
    ('Rubik', ['SV', 'OCC', 'IPR', 'OPR']),
    ('Singer1', ['IV', 'SV', 'OCC', 'OPR']),
    ('Skater', ['SV', 'DEF', 'IPR', 'OPR']),
    ('Skater2', ['SV', 'DEF', 'FM', 'IPR', 'OPR']),
    ('Subway', ['OCC', 'DEF', 'BC']),
    ('Suv', ['OCC', 'IPR', 'OV']),
    ('Tiger1', ['IV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR']),
    ('Toy', ['SV', 'FM', 'IPR', 'OPR']),
    ('Trans', ['IV', 'SV', 'OCC', 'DEF']),
    ('Twinnings', ['SV', 'OPR']),
    ('Vase', ['SV', 'FM', 'IPR']),
]

TC128_SEQUENCES_ATTRIBUTES = [
    ('Airport_ce', ['SV', 'OCC']),
    ('Baby_ce', ['SV', 'OCC', 'DEF', 'OPR']),
    ('Badminton_ce1', ['DEF', 'MB', 'OPR']),
    ('Badminton_ce2', ['DEF', 'MB', 'OPR']),
    ('Ball_ce1', ['OCC', 'MB', 'IPR', 'OPR', 'FM']),
    ('Ball_ce2', ['IV', 'OCC', 'MB', 'OV', 'FM']),
    ('Ball_ce3', ['OV', 'SV', 'FM']),
    ('Ball_ce4', ['FM', 'OCC', 'OPR', 'OV']),
    ('Basketball', ['IV', 'OPR', 'OCC', 'DEF', 'BC']),
    ('Basketball_ce1', ['IV', 'SV', 'OCC', 'DEF', 'MB', 'OPR', 'BC']),
    ('Basketball_ce2', ['IV', 'OCC', 'DEF', 'OPR', 'BC']),
    ('Basketball_ce3', ['SV', 'OCC', 'DEF', 'MB', 'OPR', 'BC']),
    ('Bee_ce', ['BC', 'LR']),
    ('Bicycle', ['IV', 'SV', 'BC']),
    ('Biker', ['SV', 'MB', 'FM', 'OPR']),
    ('Bikeshow_ce', ['SV', 'DEF', 'FM', 'IPR', 'OPR']),
    ('Bike_ce1', ['IV', 'SV', 'FM', 'IPR', 'BC']),
    ('Bike_ce2', ['SV', 'IPR', 'LR']),
    ('Bird', ['OCC', 'FM', 'OPR']),
    ('Board', ['SV', 'MB', 'FM', 'OV', 'OPR']),
    ('Boat_ce1', ['SV', 'OCC', 'IPR', 'OPR']),
    ('Boat_ce2', ['SV', 'OCC', 'FM', 'IPR', 'OPR', 'LR']),
    ('Bolt', ['OPR', 'OCC', 'DEF', 'IPR']),
    ('Boy', ['OPR', 'SV', 'MB', 'FM', 'IPR']),
    ('Busstation_ce1', ['OCC', 'BC']),
    ('Busstation_ce2', ['OCC', 'IPR', 'OPR', 'BC']),
    ('Carchasing_ce1', ['IV', 'SV', 'OCC', 'FM', 'IPR']),
    ('Carchasing_ce3', []),
    ('Carchasing_ce4', ['SV']),
    ('CarDark', ['IV', 'BC']),
    ('CarScale', ['OPR', 'SV', 'OCC', 'FM', 'IPR']),
    ('Charger_ce', ['IV', 'OCC', 'MB', 'IPR', 'OPR', 'OV', 'SV', 'FM']),
    ('Coke', ['IV', 'OPR', 'OCC', 'FM', 'IPR']),
    ('Couple', ['OPR', 'SV', 'DEF', 'FM', 'BC']),
    ('Crossing', ['SV', 'DEF', 'BC']),
    ('Cup', ['BC']),
    ('Cup_ce', ['MB', 'IPR', 'OPR', 'FM']),
    ('David', ['IV', 'OPR', 'SV', 'OCC', 'DEF', 'MB', 'IPR']),
    ('David3', ['OPR', 'OCC', 'DEF', 'BC']),
    ('Deer', ['MB', 'FM', 'IPR', 'BC']),
    ('Diving', ['SV', 'DEF', 'MB']),
    ('Doll', ['IV', 'OPR', 'SV', 'OCC', 'IPR', 'FM']),
    ('Eagle_ce', ['SV', 'IPR', 'OPR', 'BC']),
    ('Electricalbike_ce', ['SV', 'OCC']),
    ('FaceOcc1', ['OCC']),
    ('Face_ce', ['SV', 'OCC', 'IPR', 'OPR', 'BC']),
    ('Face_ce2', ['IV', 'OCC', 'MB', 'IPR', 'OPR', 'FM']),
    ('Fish_ce1', ['OCC', 'IPR', 'OPR', 'SV']),
    ('Fish_ce2', ['OCC', 'IPR', 'OPR', 'SV']),
    ('Football1', ['OPR', 'IPR', 'BC']),
    ('Girl', ['OPR', 'SV', 'OCC', 'IPR']),
    ('Girlmov', ['OCC', 'MB', 'OPR']),
    ('Guitar_ce1', ['FM']),
    ('Guitar_ce2', ['IV', 'FM', 'IPR', 'OPR']),
    ('Gym', ['SV', 'DEF', 'IPR', 'OPR']),
    ('Hand', ['IV', 'SV']),
    ('Hand_ce1', ['FM', 'DEF', 'MB', 'BC']),
    ('Hand_ce2', ['MB', 'OV', 'FM']),
    ('Hurdle_ce1', ['DEF', 'MB', 'BC']),
    ('Hurdle_ce2', ['DEF', 'FM', 'BC']),
    ('Iceskater', ['SV', 'IPR', 'OPR']),
    ('Ironman', ['IV', 'OPR', 'OCC', 'MB', 'FM', 'IPR', 'OV', 'BC', 'SV']),
    ('Jogging_1', ['OCC', 'DEF', 'OPR']),
    ('Jogging_2', ['OCC', 'DEF', 'OPR']),
    ('Juice', ['SV']),
    ('Kite_ce1', ['OCC', 'IPR', 'BC']),
    ('Kite_ce2', ['SV', 'OCC', 'IPR', 'BC']),
    ('Kite_ce3', ['FM', 'IPR', 'BC']),
    ('Kobe_ce', ['SV', 'OCC', 'DEF', 'MB', 'IPR', 'OPR', 'BC']),
    ('Lemming', ['IV', 'OPR', 'SV', 'OCC', 'FM', 'OV']),
    ('Liquor', ['IV', 'OPR', 'SV', 'OCC', 'MB', 'FM', 'OV', 'BC']),
    ('Logo_ce', ['SV', 'IPR']),
    ('Matrix', ['IV', 'OPR', 'SV', 'OCC', 'FM', 'IPR', 'BC']),
    ('Messi_ce', ['SV', 'OCC', 'DEF', 'MB', 'IPR', 'BC']),
    ('Michaeljackson_ce', ['IV', 'DEF', 'FM', 'IPR', 'OPR']),
    ('Microphone_ce1', ['OCC', 'FM']),
    ('Microphone_ce2', ['OPR', 'LR']),
    ('Motorbike_ce', ['IV', 'OCC', 'BC']),
    ('MotorRolling', ['IV', 'SV', 'MB', 'FM', 'IPR', 'BC']),
    ('MountainBike', ['OPR', 'IPR', 'BC']),
    ('Panda', ['SV', 'OCC', 'IPR', 'LR']),
    ('Plane_ce2', ['SV', 'FM', 'IPR', 'OPR']),
    ('Plate_ce1', ['SV', 'LR']),
    ('Plate_ce2', ['LR']),
    ('Pool_ce1', ['LR']),
    ('Pool_ce2', ['LR']),
    ('Pool_ce3', ['MB', 'BC', 'LR']),
    ('Railwaystation_ce', ['OCC', 'IPR', 'BC']),
    ('Ring_ce', ['BC', 'LR']),
    ('Sailor_ce', ['FM']),
    ('Shaking', ['IV', 'OPR', 'IPR', 'BC']),
    ('Singer1', ['IV', 'OPR', 'SV', 'OCC']),
    ('Singer2', ['IV', 'OPR', 'DEF', 'IPR', 'BC']),
    ('Singer_ce1', ['IV', 'DEF', 'FM', 'BC']),
    ('Singer_ce2', ['IV', 'SV', 'DEF', 'OPR', 'BC']),
    ('Skating1', ['IV', 'OPR', 'SV', 'OCC', 'DEF', 'BC']),
    ('Skating2', ['SV', 'DEF', 'FM', 'OPR']),
    ('Skating_ce1', ['SV', 'OCC', 'DEF', 'FM', 'IPR', 'OPR']),
    ('Skating_ce2', ['SV', 'MB', 'FM', 'IPR', 'OPR']),
    ('Skiing', ['IV', 'OPR', 'SV', 'DEF', 'IPR', 'FM', 'LR']),
    ('Skiing_ce', ['SV', 'OCC', 'FM']),
    ('Skyjumping_ce', ['IV', 'SV', 'DEF', 'FM', 'OPR']),
    ('Soccer', ['IV', 'OPR', 'SV', 'OCC', 'MB', 'FM', 'IPR', 'BC']),
    ('Spiderman_ce', ['OCC', 'SV', 'FM', 'IPR', 'OPR']),
    ('Subway', ['OCC', 'DEF', 'BC']),
    ('Suitcase_ce', ['IV', 'OCC', 'BC']),
    ('Sunshade', ['IV']),
    ('SuperMario_ce', ['OV', 'LR']),
    ('Surf_ce1', ['OCC', 'SV', 'FM', 'IPR', 'OPR']),
    ('Surf_ce2', ['OCC', 'SV', 'FM', 'IPR', 'OPR']),
    ('Surf_ce3', ['OCC', 'FM', 'IPR', 'OPR']),
    ('Surf_ce4', ['FM', 'IPR', 'OPR']),
    ('TableTennis_ce', ['MB', 'BC', 'LR']),
    ('TennisBall_ce', ['OCC', 'MB', 'FM', 'LR']),
    ('Tennis_ce1', ['SV', 'MB', 'IPR', 'OPR']),
    ('Tennis_ce2', ['MB', 'FM', 'IPR', 'OPR']),
    ('Tennis_ce3', ['OPR']),
    ('Thunder_ce', []),
    ('Tiger1', ['IV', 'OPR', 'OCC', 'DEF', 'MB', 'FM', 'IPR']),
    ('Tiger2', ['IV', 'OPR', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OV']),
    ('Torus', ['OPR']),
    ('Toyplane_ce', ['SV', 'OCC', 'FM', 'OPR']),
    ('Trellis', ['IV', 'OPR', 'SV', 'IPR', 'BC']),
    ('Walking', ['SV', 'OCC', 'DEF', 'LR']),
    ('Walking2', ['SV', 'OCC']),
    ('Woman', ['IV', 'OPR', 'SV', 'OCC', 'DEF', 'MB', 'FM']),
    ('Yo-yos_ce1', ['MB', 'OV', 'LR', 'FM']),
    ('Yo-yos_ce2', ['MB', 'OV', 'LR', 'FM']),
    ('Yo-yos_ce3', ['MB', 'FM', 'OV', 'SV'])
]
"""
Human3, 0.601, 0.934, 0.000, 0.587
Surfer, 0.607, 0.875, 0.490, -0.009
Boy, 0.797, 0.997, 0.434, -0.013
Deer, 0.701, 1.000, 0.385, -0.033
Jumping, 0.659, 0.962, 0.384, 0.005
Tiger1, 0.556, 0.621, 0.316, -0.019
Basketball, 0.624, 0.961, 0.000, 0.301
ClifBar, 0.443, 0.625, 0.002, 0.230
KiteSurf, 0.658, 0.869, 0.187, 0.002
BlurOwl, 0.836, 0.989, 0.180, 0.000
Suv, 0.646, 0.913, 0.000, 0.151
DragonBaby, 0.587, 0.761, 0.013, 0.132
Human4, 0.490, 0.814, 0.000, 0.106
Trellis, 0.618, 0.812, 0.087, 0.049
CarDark, 0.580, 0.707, 0.000, 0.084
Human7, 0.735, 0.976, 0.079, -0.006
Girl, 0.690, 1.000, 0.000, 0.056
Coke, 0.510, 0.746, 0.000, 0.048
BlurCar3, 0.836, 1.000, 0.033, 0.037
Woman, 0.583, 0.988, 0.000, 0.039
Girl2, 0.600, 0.833, 0.038, 0.023
BlurCar2, 0.839, 1.000, 0.034, 0.024
FaceOcc1, 0.740, 0.924, 0.000, 0.030
Car1, 0.804, 1.000, 0.000, 0.027
Singer1, 0.697, 0.946, 0.000, 0.026
Crowds, 0.698, 1.000, 0.000, 0.022
Dancer2, 0.761, 1.000, 0.000, 0.021
CarScale, 0.652, 0.706, 0.000, 0.021
Sylvester, 0.703, 0.949, 0.000, 0.018
Rubik, 0.709, 0.829, 0.000, 0.018
Couple, 0.717, 1.000, 0.012, 0.013
David, 0.774, 0.992, 0.000, 0.013
Fish, 0.755, 0.998, 0.000, 0.012
Jogging_1, 0.738, 0.977, -0.001, 0.011
BlurCar1, 0.838, 0.996, 0.009, 0.010
BlurFace, 0.836, 1.000, 0.005, 0.010
Human2, 0.770, 0.948, 0.000, 0.010
Car4, 0.716, 0.910, 0.005, 0.009
Shaking, 0.719, 0.995, 0.000, 0.007
Jogging_2, 0.751, 0.971, 0.000, 0.007
Dancer, 0.779, 1.000, 0.000, 0.007
FaceOcc2, 0.704, 0.883, 0.000, 0.006
BlurBody, 0.722, 0.946, 0.000, 0.005
Car24, 0.841, 1.000, 0.000, 0.003
Gym, 0.536, 0.996, 0.000, 0.002
BlurCar4, 0.847, 1.000, 0.000, 0.002
Crossing, 0.786, 1.000, 0.000, -0.000
Dog1, 0.813, 1.000, 0.000, -0.002
Skater, 0.650, 0.994, 0.000, -0.004
Mhyang, 0.788, 1.000, 0.000, -0.004
Walking, 0.728, 1.000, 0.000, -0.007
Freeman3, 0.759, 1.000, 0.000, -0.009
RedTeam, 0.667, 1.000, 0.000, -0.011
Skating2_2, 0.512, 0.450, 0.000, -0.012
Human5, 0.737, 0.992, -0.013, 0.000
Bird2, 0.664, 0.808, -0.001, -0.013
Skater2, 0.609, 0.931, 0.000, -0.015
Subway, 0.755, 1.000, 0.000, -0.015
Twinnings, 0.545, 0.828, 0.000, -0.024
MountainBike, 0.695, 1.000, 0.000, -0.024
Bolt2, 0.627, 0.904, 0.000, -0.025
Vase, 0.555, 0.775, 0.000, -0.038
Human6, 0.700, 0.907, 0.000, -0.048
Panda, 0.583, 1.000, 0.000, -0.050
FleetFace, 0.541, 0.620, 0.000, -0.053
Man, 0.628, 1.000, 0.000, -0.067
Human9, 0.652, 0.885, 0.019, -0.107
Lemming, 0.552, 0.594, 0.000, -0.110
Dudek, 0.534, 0.599, 0.000, -0.224
David2, 0.569, 0.762, 0.000, -0.228
Biker, 0.429, 0.627, 0.085, -0.233
Car2, 0.618, 0.775, -0.202, -0.214
David3, 0.440, 0.567, 0.000, -0.277
Football, 0.408, 0.517, -0.001, -0.306
"""
IZTECH15OTB_SEQUENCES_ATTRIBUTES = [
    ('Human3', ['SV', 'OCC', 'DEF', 'OPR', 'BC']),
    ('Surfer', ['SV', 'FM', 'IPR', 'OPR', 'LR']),
    ('Boy', ['SV', 'MB', 'FM', 'IPR', 'OPR']),
    ('Deer', ['MB', 'FM', 'IPR', 'BC', 'LR']),
    ('Jumping', ['MB', 'FM']),
    ('Tiger1', ['IV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR']),
    ('Basketball', ['IV', 'OPR', 'OCC', 'DEF', 'BC']),
    ('KiteSurf', ['IV', 'OCC', 'IPR', 'OPR']),
    ('Coke', ['IV', 'OCC', 'FM', 'IPR', 'OPR', 'BC']),
    ('Suv', ['OCC', 'IPR', 'OV']),
    ('DragonBaby', ['SV', 'OCC', 'MB', 'FM', 'IPR', 'OPR', 'OV']),
    ('Human4', ['IV', 'SV', 'OCC', 'DEF']),
    ('Trellis', ['IV', 'SV', 'IPR', 'OPR', 'BC']),
    ('CarDark', ['IV', 'BC']),
    ('Human7', ['IV', 'SV', 'OCC', 'DEF', 'MB', 'FM']),
]

"""
Railwaystation_ce, 0.722, 0.990, -0.005, 0.624
Pool_ce1, 0.655, 1.000, 0.619, -0.007
Pool_ce2, 0.594, 1.000, 0.575, 0.018
Messi_ce, 0.638, 0.996, 0.559, -0.008
Face_ce2, 0.597, 0.716, 0.007, 0.527
Busstation_ce1, 0.591, 0.983, -0.016, 0.509
Bike_ce2, 0.584, 1.000, 0.492, -0.005
Boy, 0.797, 0.997, 0.434, -0.013
Bicycle, 0.608, 0.970, 0.419, 0.252
Deer, 0.701, 1.000, 0.385, -0.033
Sunshade, 0.626, 0.855, 0.323, -0.126
Tiger1, 0.556, 0.621, 0.316, -0.019
Basketball, 0.624, 0.961, 0.000, 0.301
Busstation_ce2, 0.684, 0.932, 0.000, 0.284
Hurdle_ce2, 0.651, 0.964, 0.049, 0.228
Michaeljackson_ce, 0.594, 0.427, -0.002, 0.176
Torus, 0.613, 0.939, 0.165, -0.006
Badminton_ce2, 0.602, 0.970, 0.000, 0.108
Skyjumping_ce, 0.505, 0.928, 0.103, -0.003
Trellis, 0.618, 0.812, 0.087, 0.049
CarDark, 0.580, 0.707, 0.000, 0.084
Tennis_ce3, 0.545, 0.926, 0.000, 0.078
Plate_ce1, 0.726, 1.000, 0.000, 0.072
Girl, 0.690, 1.000, 0.000, 0.056
Tennis_ce1, 0.657, 1.000, 0.052, 0.025
Electricalbike_ce, 0.701, 0.998, 0.000, 0.050
Coke, 0.510, 0.746, 0.000, 0.048
Woman, 0.583, 0.988, 0.000, 0.039
Sailor_ce, 0.745, 0.988, 0.036, -0.023
FaceOcc1, 0.740, 0.924, 0.000, 0.030
Surf_ce1, 0.601, 0.465, 0.001, 0.027
Singer1, 0.697, 0.946, 0.000, 0.026
Skiing_ce, 0.700, 0.881, -0.000, 0.022
CarScale, 0.652, 0.706, 0.000, 0.021
Baby_ce, 0.552, 1.000, 0.000, 0.017
Couple, 0.717, 1.000, 0.012, 0.013
David, 0.774, 0.992, 0.000, 0.013
Jogging_1, 0.738, 0.977, -0.001, 0.011
Bee_ce, 0.679, 1.000, 0.000, 0.011
Shaking, 0.719, 0.995, 0.000, 0.007
Charger_ce, 0.663, 0.581, 0.000, 0.007
Jogging_2, 0.751, 0.971, 0.000, 0.007
Suitcase_ce, 0.615, 0.810, -0.003, 0.006
Fish_ce2, 0.600, 0.921, 0.000, 0.005
Tennis_ce2, 0.772, 1.000, 0.000, 0.003
Guitar_ce2, 0.700, 0.760, 0.000, 0.003
Plane_ce2, 0.553, 0.979, 0.002, 0.001
Carchasing_ce4, 0.662, 1.000, 0.001, 0.001
Carchasing_ce3, 0.709, 1.000, 0.000, 0.001
Kite_ce3, 0.628, 1.000, 0.000, 0.001
Surf_ce3, 0.541, 0.781, 0.000, 0.000
"""
IZTECH15TC_SEQUENCES_ATTRIBUTES = [
    ('Railwaystation_ce', ['OCC', 'IPR', 'BC']),
    ('Messi_ce', ['SV', 'OCC', 'DEF', 'MB', 'IPR', 'BC']),
    ('Face_ce2', ['IV', 'OCC', 'MB', 'IPR', 'OPR', 'FM']),
    ('Busstation_ce1', ['OCC', 'BC']),
    ('Boy', ['OPR', 'SV', 'MB', 'FM', 'IPR']),
    ('Bicycle', ['IV', 'SV', 'BC']),
    ('Deer', ['MB', 'FM', 'IPR', 'BC']),
    ('Tiger1', ['IV', 'OPR', 'OCC', 'DEF', 'MB', 'FM', 'IPR']),
    ('Basketball', ['IV', 'OPR', 'OCC', 'DEF', 'BC']),
    ('Busstation_ce2', ['OCC', 'IPR', 'OPR', 'BC']),
    ('Hurdle_ce2', ['DEF', 'FM', 'BC']),
    ('Michaeljackson_ce', ['IV', 'DEF', 'FM', 'IPR', 'OPR']),
    ('Badminton_ce2', ['DEF', 'MB', 'OPR']),
    ('Skyjumping_ce', ['IV', 'SV', 'DEF', 'FM', 'OPR']),
    ('Trellis', ['IV', 'OPR', 'SV', 'IPR', 'BC']),
]

OTB100_SEQUENCES = []
OTB100_ATTRIBUTES = []
for sequence, attributes in OTB100_SEQUENCES_ATTRIBUTES:
    OTB100_SEQUENCES.append(sequence)
    for attribute in attributes:
        OTB100_ATTRIBUTES.append(attribute)

OTB100_ATTRIBUTES = list(set(OTB100_ATTRIBUTES))
OTB100_SEQUENCES = sorted(OTB100_SEQUENCES)
OTB100_ATTRIBUTES = sorted(OTB100_ATTRIBUTES)

TC128_SEQUENCES = []
TC128_ATTRIBUTES = []
for sequence, attributes in TC128_SEQUENCES_ATTRIBUTES:
    TC128_SEQUENCES.append(sequence)
    for attribute in attributes:
        TC128_ATTRIBUTES.append(attribute)

TC128_ATTRIBUTES = list(set(TC128_ATTRIBUTES))
TC128_SEQUENCES = sorted(TC128_SEQUENCES)
TC128_ATTRIBUTES = sorted(TC128_ATTRIBUTES)

IZTECH15OTB_SEQUENCES = []
IZTECH15OTB_ATTRIBUTES = []
for sequence, attributes in IZTECH15OTB_SEQUENCES_ATTRIBUTES:
    IZTECH15OTB_SEQUENCES.append(sequence)
    for attribute in attributes:
        IZTECH15OTB_ATTRIBUTES.append(attribute)

IZTECH15OTB_ATTRIBUTES = list(set(IZTECH15OTB_ATTRIBUTES))
IZTECH15OTB_SEQUENCES = sorted(IZTECH15OTB_SEQUENCES)
IZTECH15OTB_ATTRIBUTES = sorted(IZTECH15OTB_ATTRIBUTES)

IZTECH15TC_SEQUENCES = []
IZTECH15TC_ATTRIBUTES = []
for sequence, attributes in IZTECH15TC_SEQUENCES_ATTRIBUTES:
    IZTECH15TC_SEQUENCES.append(sequence)
    for attribute in attributes:
        IZTECH15TC_ATTRIBUTES.append(attribute)

IZTECH15TC_ATTRIBUTES = list(set(IZTECH15TC_ATTRIBUTES))
IZTECH15TC_SEQUENCES = sorted(IZTECH15TC_SEQUENCES)
IZTECH15TC_ATTRIBUTES = sorted(IZTECH15TC_ATTRIBUTES)
