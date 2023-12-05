from array import array
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import seaborn as sns
from tensorflow.python.framework.errors_impl import NotFoundError

from keras.losses import Loss, MeanSquaredError, MeanAbsolutePercentageError
from keras.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.optimizers.legacy import Adam
from keras.models import load_model
from models import Residual, tf_params2pred, GRU, LSTM, Dense, Reshape, Input, EarlyStopping, Sequential, \
	LSTMCell, GRUCell, ConvLSTM2D, Feedback, ResidualFeedback, TimeDistributed, Lambda
import pickle
from constants import input_columns, output_columns, PATIENCE, MAX_NUM_EPOCHS, n_turbines
from init import DATA_SAVE_DIR
from Data import Data
from multiprocessing import Pool, cpu_count
from functools import partial

def main():
	## LOAD BASE AND SIMULATION FLORIS WIND FARM MODELS
	
	## GENERATE/LOAD TIME-SERIES OF FREE-STREAM WIND-SPEED, WIND DIRECTION, TURBINE AXIAL INDUCTION FACTORS AND YAW ANGLES
	wf_dfs = []
	for _, _, filenames in os.walk(DATA_SAVE_DIR):
		for fn in filenames:
			if 'case' not in fn:
				continue
			
			## convert wind speed and direction to vectors
			df = pd.read_csv(os.path.join(DATA_SAVE_DIR, fn), index_col=0)
			# if 'FreestreamWindSpeed' not in df.columns:
			# 	df['FreestreamWindSpeed'] = df[f'TurbineWindSpeedsSimulation_{0}']
			
			# df.rename(columns={f'TurbineWindSpeedsSimulation_{t}': f'TurbineWindSpeeds_{t}'
			#            for t in range(n_turbines)}, inplace=True)
			# df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
			# df.to_csv(os.path.join(DATA_SAVE_DIR, fn))
			
			wind_speeds = df.pop('FreestreamWindSpeed')
			# print(wind_speeds.max())
			wind_directions = df.pop('FreestreamWindDir') * np.pi / 180
			# compute freestream wind speed components where 0 deg = (from) northerly wind,
			# 90 deg (from) easterly, 180 deg (from) southerly and 270 deg (from) westerly
			df[f'FreestreamWindSpeedsY'] = wind_speeds * -np.cos(wind_directions)
			df[f'FreestreamWindSpeedsX'] = wind_speeds * -np.sin(wind_directions)
			# print(df[f'FreestreamWindSpeedsY'].max())
			# print(df[f'FreestreamWindSpeedsX'].max())
			
			wf_dfs.append(df)
	
	[col for col in wf_dfs[0].columns if 'AxIndFactors' not in col and 'Time' not in col]
	desc = wf_dfs[0][[col for col in wf_dfs[0].columns if 'AxIndFactors' not in col and 'Time' not in col]].head().transpose().sort_index()
	
	## GENERATE DATASETS
	# 70, 20, 10 raw_data split for training, validation and testing raw_data
	training_end_idx = int(0.7 * len(wf_dfs))
	val_end_idx = int(0.9 * len(wf_dfs))
	
	# X_base_columns = ['FreestreamWindSpeedsX', 'FreestreamWindSpeedsY']
	X_turbine_column_prefixes = [f'TurbineWindSpeeds_', f'YawAngles_'] #, f'AxIndFactors_']
	y_turbine_column_prefixes = [f'TurbineWindSpeeds_']
	
	# + [f'AxIndFactors_{t}' for t in downstream_turbine_indices] \
	
	## NORMALIZE DATA
	data_min = pd.concat([wf_dfs[i] for i in range(len(wf_dfs))]).min()
	data_max = pd.concat([wf_dfs[i] for i in range(len(wf_dfs))]).max()
	data_mean = pd.concat([wf_dfs[i] for i in range(len(wf_dfs))]).mean()
	data_std = pd.concat([wf_dfs[i] for i in range(len(wf_dfs))]).std()
	
	
	data_std_norm = [(wf_dfs[i] - data_mean) / data_std for i in range(len(wf_dfs))]
	data_minmax_norm = [(wf_dfs[i] - data_min) / (data_max - data_min) for i in range(len(wf_dfs))]
	
	# train_dfs = data_std_norm[:training_end_idx]
	# val_dfs = data_std_norm[training_end_idx:val_end_idx]
	# test_dfs = data_std_norm[val_end_idx:]
	train_dfs = data_minmax_norm[:training_end_idx]
	val_dfs = data_minmax_norm[training_end_idx:val_end_idx]
	test_dfs = data_minmax_norm[val_end_idx:]
	
	for data_div in ['train_dfs', 'val_dfs', 'test_dfs']:
		with open(f'./clean_data/{data_div}', 'wb') as fp:
			x = locals()[data_div]
			pickle.dump(x, fp)
	
	## INSPECT DATASET
	desc = train_dfs[0][[col for col in wf_dfs[0].columns if 'AxIndFactors' not in col and 'Time' not in col]].describe().transpose().sort_index()
	desc
	
	## GENERATE INPUT AND OUTPUT DATA WINDOWS
	
	data_obj = Data(train_dfs=train_dfs, val_dfs=val_dfs, test_dfs=test_dfs,
	                input_columns=input_columns, output_columns=output_columns,
	                input_width=60, input_step=1, io_offset=60, output_width=60, output_step=1)

	# Stack three slices, the length of the total window.
	# example_window = tf.stack([np.array(train_dfs[0][:data_obj.total_window_size]),
	#                            np.array(train_dfs[0][100:100 + data_obj.total_window_size]),
	#                            np.array(train_dfs[0][200:200 + data_obj.total_window_size])])
	#
	# example_inputs, example_labels = data_obj.split_window(example_window)
	#
	# print('All shapes are: (batch, time, features)')
	# print(f'Window shape: {example_window.shape}')
	# print(f'Inputs shape: {example_inputs.shape}')
	# print(f'Labels shape: {example_labels.shape}')
	#
	# data_obj.plot(example_inputs, example_labels)
	
	data_obj.train.element_spec
	data_obj.train_multi.element_spec
	
	# iterate over first training dataset
	for example_inputs, example_labels in data_obj.train.take(1):
		print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
		print(f'Labels shape (batch, time, features): {example_labels.shape}')
	
	for example_inputs, example_labels in data_obj.train_multi.take(1):
		print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
		print(f'Labels shape (batch, time, features): {example_labels.shape}')
		
	## PLOTTING DATA
	if False:
		# TODO Plot time-series
		plot_cols = [['FreestreamWindSpeedsX'], ['FreestreamWindSpeedsY']] \
		            + [[pf + str(i) for i in range(n_turbines)] for pf in X_turbine_column_prefixes]
		plot_features = wf_dfs[0][np.concatenate(plot_cols)]
		time_seconds = wf_dfs[0]['Time'].to_numpy()
		plot_features.index = datetime(1, 1, 1) + np.array([timedelta(seconds=s) for s in time_seconds])
		fig, axs = plt.subplots(len(plot_cols), 1, sharex=True)
		idx = slice(0, 600, 1)
		for a, ax in enumerate(axs):
			for col in plot_cols[a]:
				if col not in plot_features:
					continue
				ax.plot(time_seconds[idx], plot_features[col].iloc[idx],
				        label=f'T{col.split("_")[1]}' if "_" in col else None)
			
			if '_' in plot_cols[a][0]:
				ax.set(title=plot_cols[a][0].split('_')[0])
			# ax.legend()
			else:
				ax.set(title=plot_cols[a][0])
		axs[-1].set(xlabel='Time [s]')
		fig.savefig('./figs/time_series.png')
		fig.show()
		
		# plot histogram of freestream wind speed components
		fig, ax = plt.subplots(1, 1)
		ax.hist2d(wf_dfs[0]['FreestreamWindSpeedsX'], wf_dfs[0]['FreestreamWindSpeedsY'], bins=(50, 50))
		fig.savefig('./figs/fs_ws_xy_hist.png')
		fig.show()
		
		# Plot distribution of features
		X_norm = data_minmax_norm[0][[col for col in wf_dfs[0].columns if 'AxIndFactors' not in col and 'Time' not in col]].melt(var_name='Column', value_name='Normalized').sort_values('Column')
		fig, axs = plt.subplots(3, 1, figsize=(24, 18))
		
		for c, col_prefix in enumerate(['FreestreamWindSpeeds', 'TurbineWindSpeeds', 'YawAngles']):
			ax = sns.violinplot(x='Column', y='Normalized',
			                    data=X_norm.loc[X_norm.Column.str.contains(col_prefix), :],
			                    ax=axs[c])
			if col_prefix == 'FreestreamWindSpeeds':
				_ = ax.set_xticklabels(['X', 'Y'])
			else:
				_ = ax.set_xticklabels([f'T{i}' for i in range(n_turbines)])
			
			# _ = ax.set_xticklabels(sorted([col for col in data_std_norm[0].columns if col_prefix in col]), rotation=90)
			ax.set(title=col_prefix, ylabel='', xlabel='')
		fig.savefig('./figs/feature_dist.png')
		fig.show()
	
	## COMPILE & FIT MODELS to learn rotor effective wind speed at each downstream turbine
	
	early_stopping_cb = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')
	models = {}
	histories = {}
	train_eval = {}
	val_eval = {}
	test_eval = {}
	
	models['feedback_gru'] = Feedback(units=32, out_steps=int(data_obj.output_width // data_obj.output_step),
	                                  num_features=len(data_obj.input_columns), cell_class=GRUCell,
	                                  output_indices=[data_obj.input_columns.index(col) for col in
	                                                  data_obj.output_columns])
	
	models['feedback_lstm'] = Feedback(units=32, out_steps=int(data_obj.output_width // data_obj.output_step),
	                                   num_features=len(data_obj.input_columns), cell_class=LSTMCell,
	                                   output_indices=[data_obj.input_columns.index(col) for col in data_obj.output_columns])

	models['feedback_res_lstm'] = ResidualFeedback(units=32,
	                                               out_steps=int(data_obj.output_width // data_obj.output_step),
	                                               num_features=len(data_obj.input_columns), cell_class=LSTMCell,
	                                               output_indices=[data_obj.input_columns.index(col) for col in data_obj.output_columns])

	models['feedback_res_gru'] = ResidualFeedback(units=32,
	                                              out_steps=int(data_obj.output_width // data_obj.output_step),
	                                              num_features=len(data_obj.input_columns), cell_class=GRUCell,
	                                              output_indices=[data_obj.input_columns.index(col) for col in data_obj.output_columns])
	
	models['multi_lstm'] = Sequential([
			LSTM(32, return_sequences=True, input_shape=[None, len(input_columns)]),
			LSTM(32, return_sequences=True),
			TimeDistributed(Dense(
				len(data_obj.input_columns) * int(data_obj.output_width // data_obj.output_step))),
			Reshape((int(data_obj.input_width // data_obj.input_step),
			         int(data_obj.output_width // data_obj.output_step),
			         len(data_obj.input_columns))),
			Lambda(lambda x: x[:, :, :,
		                 data_obj.input_columns.index(data_obj.output_columns[0]):(data_obj.input_columns.index(data_obj.output_columns[-1]) + 1)]),
			Reshape((int(data_obj.input_width // data_obj.input_step) *
			         int(data_obj.output_width // data_obj.output_step),
		         len(data_obj.output_columns)))
		])
	models['multi_lstm'].summary()
	
	models['multi_gru'] = Sequential([
			GRU(32, return_sequences=True, input_shape=[None, len(input_columns)]),
			GRU(32, return_sequences=True),
			TimeDistributed(Dense(
				len(data_obj.input_columns) * int(data_obj.output_width // data_obj.output_step))),
			Reshape((int(data_obj.input_width // data_obj.input_step),
			         int(data_obj.output_width // data_obj.output_step),
			         len(data_obj.input_columns))),
			Lambda(lambda x: x[:, :, :,
			                 data_obj.input_columns.index(data_obj.output_columns[0]):(
					                 data_obj.input_columns.index(data_obj.output_columns[-1]) + 1)]),
			Reshape((int(data_obj.input_width // data_obj.input_step) *
			         int(data_obj.output_width // data_obj.output_step),
			         len(data_obj.output_columns)))
		])
	models['multi_gru'].summary()
	
	# models['multi_res_lstm'] = Residual(
	# 	Sequential([
	# 		LSTM(32, return_sequences=True, input_shape=[None, len(input_columns)]),
	# 		LSTM(32, return_sequences=True),
	# 		TimeDistributed(Dense(
	# 			len(data_obj.input_columns) * int(data_obj.output_width // data_obj.output_step))),
	# 		Reshape((int(data_obj.input_width // data_obj.input_step) *
	# 		         int(data_obj.output_width // data_obj.output_step),
	# 		         len(data_obj.input_columns)))
	# 	]), output_indices=[data_obj.input_columns.index(col) for col in data_obj.output_columns] # indices of output columns in input columns
	# )
	#
	# models['multi_res_gru'] = Residual(
	# 	Sequential([
	# 		GRU(32, return_sequences=True, input_shape=[None, len(input_columns)]),
	# 		GRU(32, return_sequences=True),
	# 		TimeDistributed(Dense(
	# 			len(data_obj.input_columns) * int(data_obj.output_width // data_obj.output_step))),
	# 		Reshape((int(data_obj.input_width // data_obj.input_step) *
	# 		         int(data_obj.output_width // data_obj.output_step),
	# 		         len(data_obj.input_columns))),
	# 	]), output_indices=[data_obj.input_columns.index(col) for col in data_obj.output_columns] # indices of output columns in input columns
	# )
	
	# models['multi_conv2d_lstm'] = Sequential([
	# 	ConvLSTM2D(32, return_sequences=False),
	# 	Dense(int(data_obj.output_width // data_obj.output_step) * len(output_columns),
	# 	      kernel_initializer=tf.initializers.zeros()),
	# 	Reshape([int(data_obj.output_width // data_obj.output_step), len(data_obj.output_columns)])
	# ])
	
	# Params2Pred_Layer = Lambda(lambda x: tf_params2pred, output_shape=(len(data_obj.output_columns),))
	
	# test_X, test_y = data_obj.train.as_numpy_iterator().next()
	# test_params = og_params + [0.001]
	# Params2Pred_Layer([test_params, test_X])
	
	# input_params = Variable((N_PARAMS + 1,))
	# input_layer = Input(shape=data_obj.train.element_spec[0].shape[1:])
	# models['pred_lstm'] = LSTM(32, return_sequences=False)(input_layer)
	# models['pred_lstm'] = Dense(units=(N_PARAMS + 1))(models['pred_lstm'])
	# models['pred_lstm'] = Params2Pred_Layer([models['pred_lstm'], input_layer])
	# models['pred_lstm'] = Model(input_layer, models['pred_lstm'])
	# models['pred_lstm'].compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=MeanAbsolutePercentageError())
	# models['pred_lstm'].summary()
	# histories['pred_lstm'] = models['pred_lstm'].fit(data_obj.train, epochs=MAX_NUM_EPOCHS,
	#                                                  validation_data=data_obj.val,
	#                                                  callbacks=[early_stopping_cb]).history
	
	# PARALLEL = False
	#
	# if PARALLEL:
	# 	with Pool(cpu_count()) as p:
	# 		res = p.starmap(partial(train_model, data_obj), list(models.items()))
	# 		for key, (history_tmp, train_eval_tmp, val_eval_tmp, test_eval_tmp) in zip(models, res):
	# 			histories[key] = history_tmp
	# 			train_eval[key] = train_eval_tmp
	# 			val_eval[key] = val_eval_tmp
	# 			test_eval[key] = test_eval_tmp
	# 			data_obj.plot(example_inputs, example_labels, model=models[key], model_name=key)
	#
	# else:
	
	for dirname in ['model_weights', 'histories', 'evaluations']:
		if not os.path.exists(f'./{dirname}'):
			os.makedirs(f'./{dirname}')
		
	for model_key, model in models.items():
		history_tmp = train_model(data_obj, model_key, model)
		histories[model_key] = history_tmp
		
	# with Pool(cpu_count()) as p:
	# 	res = p.starmap(partial(evaluate_model, data_obj), zip(list(models.items()), list(models.keys())))
	
	for model_key, model in models.items():
		if model_key not in ['feedback_gru']:
			continue
		
		# if model_key not in ['feedback_lstm']:
		# 	continue
		# if model_key not in ['feedback_gru', 'feedback_lstm']:
		# 	continue
			
		# if model_key not in ['feedback_res_gru', 'feedback_res_lstm']:
		# 	continue

		# if model_key not in ['multi_gru', 'multi_lstm']:
		# 	continue
		
		if not os.path.exists(f'./evaluations/{model_key}'):
			train_eval_tmp, val_eval_tmp, test_eval_tmp = evaluate_model(data_obj, model, model_key)
			train_eval[model_key] = train_eval_tmp
			val_eval[model_key] = val_eval_tmp
			test_eval[model_key] = test_eval_tmp
	
	all_evals = {}
	for model_key, model in models.items():
		
		if os.path.exists(f'./evaluations/{model_key}'):
			with open(f'./evaluations/{model_key}', 'rb') as fp:
				evals = pickle.load(fp)
		else:
			evals = {}
		
		if model_key in train_eval and model_key in val_eval and model_key in test_eval:
			evals['train'] = train_eval[model_key]
			evals['test'] = test_eval[model_key]
			evals['val'] = val_eval[model_key]
			
		with open(f'./evaluations/{model_key}', 'wb') as fp:
			pickle.dump(evals, fp)
		all_evals[model_key] = evals
	
	## PLOTTING PREDICTIIONS
	if False:
		for key, model in models.items():
			print(f'Plotting time-series from {key}')
			data_obj.plot(example_inputs, example_labels, model=model, model_name=key)
	
	## PLOTTING LOSSES AND METRICS
	model_names = [n.replace('feedback_res_', 'Autoregressive Residual ')
	               .replace('feedback_', 'Autoregressive Absolute ')
	               .replace('multi_', 'Multi-Step Absolute ')
	               .replace('lstm', 'LSTM')
	               .replace('gru', 'GRU') for n in models]
	metric_labels = ['MSE [(m/s)^2]', 'MAPE [%]']
	n_metrics = len(metric_labels)
	
	if False:
		x = np.arange(len(models))
		width = 0.25
		
		fig, axs = plt.subplots(n_metrics + 1, 1, sharex=True, figsize=(54, 36)) # add room for legend
		
		for m in range(n_metrics):
			# for mm, model in enumerate(models):
			train_err = [all_evals[model]['train'][m] for model in models]
			val_err = [all_evals[model]['val'][m] for model in models]
			test_err = [all_evals[model]['test'][m] for model in models]
			
			if m == 0:
				axs[m + 1].bar(x, train_err, width, label='Train')
				axs[m + 1].bar(x + (width), val_err, width, label='Validation')
				axs[m + 1].bar(x + (width * 2), test_err, width, label='Test')
			else:
				axs[m + 1].bar(x, train_err, width)
				axs[m + 1].bar(x + (width), val_err, width)
				axs[m + 1].bar(x + (width * 2), test_err, width)
			axs[m + 1].set_ylabel(metric_labels[m], rotation=45)
			axs[m + 1].grid(True)
		axs[-1].set_xticks(x + width, model_names, rotation=45)
		# axs[-1].set_ylim((1, 2000))
		axs[0].set_axis_off()
		axs[0].set_axis_off()
		fig.tight_layout()
		fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.8), ncol=3)
		fig.show()
		
		fig.savefig('./figs/model_evals.png')
	
	## PLOTTING LEARNING CURVES
	if False:
		fig, axs = plt.subplots(n_metrics + 1, 2, sharex=True, figsize=(54, 24))  # making room for legend
		for m in range(n_metrics):
			for mm, model in enumerate(models):
				# metric = models[model].metrics_names[m]
				
				y_train = list(filter(lambda x: 'val' not in x[0], histories[model].items()))[m][1]
				y_val = list(filter(lambda x: 'val' in x[0], histories[model].items()))[m][1]
				x = np.arange(len(y_train))
				# y_train = histories[model][f'{metric}']
				# y_val = histories[model][f'val_{metric}']
				
				if m == 1:
					axs[m + 1, 0].plot(x, y_train, label=f'{model_names[mm]}')
				else:
					axs[m + 1, 0].plot(x, y_train)
				axs[m + 1, 1].plot(x, y_val)
		
		for m in range(n_metrics):
			axs[m + 1, 0].get_shared_y_axes().join(axs[m + 1, 0], axs[m + 1, 1])
			axs[m + 1, 0].get_shared_x_axes().join(axs[m + 1, 0], axs[m + 1, 1])
		
		axs[1, 0].set(title='Training Data Learning Curve')
		axs[1, 1].set(title='Validation Data Learning Curve')
		axs[1, 0].set_ylabel(metric_labels[0], rotation=45)
		axs[2, 0].set_ylabel(metric_labels[1], rotation=45)
		# pos = axs[0, 0].get_position()
		# axs[0, 0].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
		
		axs[2, 0].set(xlabel='Epoch', xticks=x)
		axs[2, 1].set(xlabel='Epoch', xticks=x)
		fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3)
		axs[0, 0].set_axis_off()
		axs[0, 1].set_axis_off()
		# fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), borderaxespad=0.)
		fig.tight_layout()
		fig.show()
		fig.savefig('./figs/learning_curve.png')
	
def LastTimeStep_MeanSquaredError(y_train, y_pred):
	return mean_squared_error(y_train[:, -1, -1, :], y_pred[:, -1, :])

def LastTimeStep_MeanAbsolutePercentageError(y_train, y_pred):
	return mean_absolute_percentage_error(y_train[:, -1, -1, :], y_pred[:, -1, :])

def train_model(data_obj, key, model):
	if 'multi' in key:
		model.compile(loss=LastTimeStep_MeanSquaredError, optimizer=Adam(),
		              metrics=LastTimeStep_MeanAbsolutePercentageError)
	else:
		model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=MeanAbsolutePercentageError())
	
	checkpoint_path = f'./model_weights/{key}_ModelWeights'
	if os.path.exists(f'{checkpoint_path}.index') and os.path.exists(f'./histories/{key}_History'):
		print(f'Loading {key} model')
		# load_model(f'./models/{key}_Model')
		model.load_weights(checkpoint_path)

		with open(f'./histories/{key}_History', 'rb') as fp:
			history = pickle.load(fp)
	else:
		print(f'Fitting {key} model')
		# Create a callback that saves the model's weights
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
		                                                 save_weights_only=True,
		                                                 verbose=1)
		
		if 'multi' in key:
			history = model.fit(data_obj.train_multi, epochs=MAX_NUM_EPOCHS,
			                    validation_data=data_obj.val_multi,
			                    callbacks=[cp_callback]).history
		else:
			history = model.fit(data_obj.train, epochs=MAX_NUM_EPOCHS,
			                    validation_data=data_obj.val,
			                    callbacks=[cp_callback]).history
		with open(f'./histories/{key}_History', 'wb') as fp:
			pickle.dump(history, fp)
	
	return history

def evaluate_model(data_obj, model, model_key):
	if 'multi' in model_key:
		train_eval = model.evaluate(data_obj.train_multi)
		val_eval = model.evaluate(data_obj.val_multi)
		test_eval = model.evaluate(data_obj.test_multi)
	else:
		train_eval = model.evaluate(data_obj.train)
		val_eval = model.evaluate(data_obj.val)
		test_eval = model.evaluate(data_obj.test)
	return train_eval, val_eval, test_eval

if __name__ == '__main__':
	main()