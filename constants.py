from floris import tools as wfct
from init import WAKE_FIELD_CONFIG
from functools import reduce

MAX_NUM_EPOCHS = 3
PATIENCE = 2

fi_sim = wfct.floris_interface.FlorisInterface(WAKE_FIELD_CONFIG["floris_input_file"])
og_config = fi_sim.floris.as_dict()
TUNABLE_MODEL_TYPES = ['deflection_model', 'turbulence_model', 'velocity_model']
TUNABLE_MODEL_PARAMS = {'deflection_model': list(
	og_config['wake']['wake_deflection_parameters'][og_config['wake']['model_strings']['deflection_model']].keys()),
	'turbulence_model': list(og_config['wake']['wake_turbulence_parameters'][
		                         og_config['wake']['model_strings']['turbulence_model']].keys()),
	'velocity_model': list(og_config['wake']['wake_velocity_parameters'][
		                       og_config['wake']['model_strings']['velocity_model']].keys())}
og_params = [
	og_config['wake'][f"wake_{model_type.split('_')[0]}_parameters"][og_config['wake']['model_strings'][model_type]][k]
	for model_type in TUNABLE_MODEL_PARAMS for k in TUNABLE_MODEL_PARAMS[model_type]]
og_params = reduce(lambda acc, x: acc + x if type(x) is list else acc + [x], og_params, [])

N_PARAMS = sum(len(l) for l in TUNABLE_MODEL_PARAMS.values())
MODEL_TYPE_OUTPUT_MAPPING = {}
max_param_idx = 0
for model_type, model_name in og_config['wake']['model_strings'].items():
	param_key = f'wake_{model_type.split("_")[0]}_parameters'
	if param_key in og_config['wake']:
		# num_params = len(og_config['wake'][param_key][model_name])
		num_params = reduce(lambda acc, x: acc + (len(x) if type(x) is list else 1),
		                    list(og_config['wake'][param_key][model_name].values()), 0)
		param_names = reduce(lambda acc, x: acc + (len(x[1]) * [x[0]] if type(x[1]) is list else [x[0]]),
		                     list(og_config['wake'][param_key][model_name].items()), [])
		
		# num_params = len(TUNABLE_MODEL_PARAMS[model_type])
		MODEL_TYPE_OUTPUT_MAPPING[model_type] = [(param_names[i], max_param_idx + i) for i in range(num_params)]
		max_param_idx = MODEL_TYPE_OUTPUT_MAPPING[model_type][-1][1] + 1

# Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
n_turbines = fi_sim.floris.farm.n_turbines
max_downstream_dist = max(fi_sim.floris.farm.coordinates, key=lambda vec: vec.x1).x1
min_downstream_dist = min(fi_sim.floris.farm.coordinates, key=lambda vec: vec.x1).x1
# exclude most downstream turbine
upstream_turbine_indices = [t for t in range(n_turbines) if
                            fi_sim.floris.farm.coordinates[t].x1 < max_downstream_dist]
# exclude most upstream turbine
downstream_turbine_indices = [t for t in range(n_turbines) if
                              fi_sim.floris.farm.coordinates[t].x1 > min_downstream_dist]
all_turbine_indices = list(range(n_turbines))
input_columns = [f'FreestreamWindSpeedsX', f'FreestreamWindSpeedsY'] \
                + [f'YawAngles_{t}' for t in all_turbine_indices] \
                + [f'TurbineWindSpeeds_{t}' for t in all_turbine_indices]
output_columns = [f'TurbineWindSpeeds_{t}' for t in downstream_turbine_indices]

yaw_angle_cols = [f"YawAngles_{t}" for t in range(n_turbines)]
yaw_angle_idx = [input_columns.index(f"YawAngles_{t}") for t in range(n_turbines)]