from floris.tools.floris_interface import FlorisInterface as DynFlorisInterface

def generate_model(floris_dir):
	## GET SYSTEM INFORMATION
	fi = DynFlorisInterface(floris_dir)
	n_turbines = len(fi.floris.farm.turbines)
	fi.max_downstream_dist = max(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
	fi.min_downstream_dist = min(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
	# exclude most downstream turbine
	fi.turbine_indices = list(range(n_turbines))
	fi.upstream_turbine_indices = [t for t in range(n_turbines)
	                                      if fi.floris.farm.turbine_map.coords[t].x1
	                                      < fi.max_downstream_dist]
	fi.downstream_turbine_indices = [t for t in range(n_turbines)
	                                        if fi.floris.farm.turbine_map.coords[t].x1
	                                        > fi.min_downstream_dist]
	
	return fi