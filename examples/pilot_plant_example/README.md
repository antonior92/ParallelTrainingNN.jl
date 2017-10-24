# Example 1: Data from a Pilot Plant

Directory containing real collected data from experiments performed on LabVolt Level Process Station (model 3503-MO) and code used to model this system.

```
example1
└─── README.md
└─── pilot_plant.ipynb
└─── calibration_data
|     └─── calibration_data.csv
└─── processed_data
|     └─── training_set.csv
|     └─── test_set.csv
|     └─── static_curve.csv
└─── raw_data
      └─── training_set.csv
      └─── test_set.csv
      └─── static_curve.csv
```

## Description

-  ``pilot_plant.ipynb``: Juyter notebook containing code modeling the system using the provided data.
-  ``processed_data``: Folder containing calibration data.
-  ``calibration_set.csv``: Data used for sensor calibration. For different watter levels gives the pressure on the sensor display (mmH2O), the water column height (mm) and the output voltage (volts)
-  ``processed_data``: Folder containing processed data ready for identification. The sample rate was decimated and the output scale adjusted according to the calibration curve.
    - ``training_set.csv``: Two hours of sensor data for a pseudorandom input. Sample rate was decimated and output scale adjusted according to the calibration curve. Input in volts and output in milimeters.
    - ``test_set.csv``: One hours of sensor data for a pseudorandom input. Sample rate was decimated and the output scale was adjusted according to the calibration curve. Input in volts and output in milimeters.
	- ``static_curve.csv``: Static curve of the system. For different input levels it was applied an constant input and the correspondent steady state output annotated. Input in volts and output in milimeters.
- ``raw_data``: Folder containing raw data.
    - ``training_set.csv``: Two hours of sensor data for a pseudorandom input. Undecimated and uncalibrated. All units are in volts.
    - ``test_set.csv``: One hours of sensor data for a pseudorandom input. Undecimated and uncalibrated. All units are in volts.
    - ``static_curve.csv``: Static curve of the system. For different input levels it was applied an constant input and the correspondent steady state output annotated. Input in volts and output in mmH2O.
