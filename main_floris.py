from array import array
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import tensorflow as tf
import keras
from keras import layers
from floris import tools as wfct
from datetime import datetime, timedelta

from helpers import generate_model
from init import *

def main():
	## LOAD BASE AND SIMULATION FLORIS WIND FARM MODELS
	
	# Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
	fi_sim = wfct.floris_interface.FlorisInterface(WAKE_FIELD_CONFIG["floris_input_file"])
	og_config = fi_sim.floris.as_dict()
	n_turbines = fi_sim.floris.farm.n_turbines
	max_downstream_dist = max(fi_sim.floris.farm.coordinates, key=lambda vec: vec.x1).x1
	min_downstream_dist = min(fi_sim.floris.farm.coordinates, key=lambda vec: vec.x1).x1
	# exclude most downstream turbine
	upstream_turbine_indices = [t for t in range(n_turbines) if
	                            fi_sim.floris.farm.coordinates[t].x1 < max_downstream_dist]
	# exclude most upstream turbine
	downstream_turbine_indices = [t for t in range(n_turbines) if
	                              fi_sim.floris.farm.coordinates[t].x1 > min_downstream_dist]
	
	## GENERATE/LOAD TIME-SERIES OF FREE-STREAM WIND-SPEED, WIND DIRECTION, TURBINE AXIAL INDUCTION FACTORS AND YAW ANGLES
	wf_dfs = []
	for _, _, filenames in os.walk(DATA_SAVE_DIR):
		for fn in filenames:
			if 'case' not in fn:
				continue
			
			## convert wind speed and direction to vectors
			df = pd.read_csv(os.path.join(DATA_SAVE_DIR, fn))
			if 'FreestreamWindSpeed' not in df.columns:
				df['FreestreamWindSpeed'] = df[f'TurbineWindSpeedsSimulation_{0}']
				df.to_csv(os.path.join(DATA_SAVE_DIR, fn))
			
			wind_speeds = df.pop('FreestreamWindSpeed')
			wind_directions = df.pop('FreestreamWindDir') * np.pi / 180
			# compute freestream wind speed components where 0 deg = (from) northerly wind,
			# 90 deg (from) easterly, 180 deg (from) southerly and 270 deg (from) westerly
			df[f'FreestreamWindSpeedsY'] = wind_speeds * -np.cos(wind_directions)
			df[f'FreestreamWindSpeedsX'] = wind_speeds * -np.sin(wind_directions)
			
			wf_dfs.append(df)
	
	## Generate X_train, y_train
	training_end_idx = int(0.8 * len(wf_dfs))
	test_end_idx = int(1.0 * len(wf_dfs))
	
	X_base_columns = ['Time', 'FreestreamWindSpeedsX', 'FreestreamWindSpeedsY']
	X_turbine_column_prefixes = [f'TurbineWindSpeedsSimulation_', f'YawAngles_', f'AxIndFactors_']
	y_turbine_column_prefixes = [f'TurbineWindSpeedsSimulation_']
	X_ds_turbine_columns = [prefix + str(i) for prefix in X_turbine_column_prefixes for i in downstream_turbine_indices]
	X_us_turbine_columns = [prefix + str(i) for prefix in X_turbine_column_prefixes for i in upstream_turbine_indices]
	X_all_turbine_columns = [prefix + str(i) for prefix in X_turbine_column_prefixes for i in range(n_turbines)]
	y_all_turbine_columns = [prefix + str(i) for prefix in y_turbine_column_prefixes for i in range(n_turbines)]
	
	y = [wf_dfs[i][y_all_turbine_columns] for i in range(len(wf_dfs))]
	X = [wf_dfs[i][X_base_columns + X_all_turbine_columns] for i in range(len(wf_dfs))]
	
	y_train = y[:training_end_idx]
	X_train = X[:training_end_idx]
	y_test = y[training_end_idx:test_end_idx]
	X_test = X[training_end_idx:test_end_idx]
	# y_val = y[test_end_idx:]
	# X_val = X[test_end_idx:]
	
	## PLOT SUBSET OF DATA
	plot_cols = [['FreestreamWindSpeedsX'], ['FreestreamWindSpeedsY']] \
	            + [[pf + str(i) for i in range(n_turbines)] for pf in X_turbine_column_prefixes]
	plot_features = X[0][np.concatenate(plot_cols)]
	time_seconds = X[0]['Time'].to_numpy()
	plot_features.index = datetime(1,1,1) + np.array([timedelta(seconds=s) for s in time_seconds])
	fig, axs = plt.subplots(len(plot_cols), 1, sharex=True)
	for a, ax in enumerate(axs):
		for col in plot_cols[a]:
			ax.plot(time_seconds, plot_features[col], label=f'T{col.split("_")[1]}')
			
		if '_' in plot_cols[a][0]:
			ax.set(title=plot_cols[a][0].split('_')[0])
			ax.legend()
		else:
			ax.set(title=plot_cols[a])
	axs[-1].set(xlabel='Time [s]')
	fig.show()
	
	## INSPECT DATASET
	X[0].describe().transpose()
	
	## GENERATE COST FUNCTION - FUNCTION OF TRUE ROTOR EFFECTIVE WIND SPEEDS AND PARAMETERS OF WAKE MODEL(s) USED
	TUNABLE_MODEL_TYPES = ['deflection_model', 'turbulence_model', 'velocity_model']
	TUNABLE_MODEL_PARAMS = {'deflection_model': list(og_config['wake']['wake_deflection_parameters'][og_config['wake']['model_strings']['deflection_model']].keys()),
	                        'turbulence_model': list(og_config['wake']['wake_turbulence_parameters'][og_config['wake']['model_strings']['turbulence_model']].keys()),
	                        'velocity_model': list(og_config['wake']['wake_velocity_parameters'][og_config['wake']['model_strings']['velocity_model']].keys())}
	
	MODEL_TYPE_OUTPUT_MAPPING = {}
	max_param_idx = 0
	for model_type, model_name in og_config['wake']['model_strings'].items():
		param_key = f'wake_{model_type.split("_")[0]}_parameters'
		if param_key in og_config['wake']:
			# num_params = len(og_config['wake'][param_key][model_name])
			num_params = len(TUNABLE_MODEL_PARAMS[model_type])
			MODEL_TYPE_OUTPUT_MAPPING[model_type] = [max_param_idx + i for i in range(num_params)]
			max_param_idx = MODEL_TYPE_OUTPUT_MAPPING[model_type][-1] + 1
	def mse_loss(X, y_true, p_pred):
		config = dict(og_config)
		for model_type in TUNABLE_MODEL_TYPES:
			model_name = config['wake']['model_strings'][model_type]
			param_key = f'wake_{model_type.split("_")[0]}_parameters'
			config['wake'][param_key][model_name].update(
				{param_name: p_pred[idx] for param_name, idx in zip(TUNABLE_MODEL_PARAMS[model_type],
				                                        MODEL_TYPE_OUTPUT_MAPPING[model_type])}
			)
		
		fi = wfct.floris_interface.FlorisInterface(config)
		
		# TODO get freestream wind speed and direction from training dataset X
		fi.reinitialize(wind_speeds=[], wind_directions=[])
		fi.calculate_wake(yaw_angles=[])
		y_pred = fi_sim.turbine_effective_velocities
		return 0.5 * sum(np.linalg.norm(y_true - y_pred), 2)**0.5
	
	
	## BULD RNN/LSTM MODEL
	
	## START SIMULATION
	# GENERATE/LOAD TIME-SERIES OF ROTOR-EFFECTIVE WIND SPEEDS FOR BASE MODEL
	# TRAIN RNN/LSTM MODEL FOR X EPOCHS WITH TRAINING DATA COLLECTED UP TO NOW
    # MAKE PREDICTIONS FOR EACH TIME-STEP
    
    ## PLOT COST FUNCTION VS SIMULATION TIME-STEP
    
if __name__ == '__main__':
	main()