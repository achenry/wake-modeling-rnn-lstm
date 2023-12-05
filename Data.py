from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from functools import partial

@dataclass
class Data:
	train_dfs: List[pd.DataFrame]
	val_dfs: List[pd.DataFrame]
	test_dfs: List[pd.DataFrame]
	
	input_columns: Optional[List[str]] = None
	output_columns: Optional[List[str]] = None
	
	input_width: int = 30 * 60  # consider inputs from last 30 minutes
	input_step: int = 60  # at time intervals of 60 seconds
	io_offset: int = 10 * 60  # make predictions up to 10 minutes aheqd of input readings
	output_width: int = 10 * 60  # make predictions for 10 minutes into the future
	output_step: int = 60  # make a prediction every minute (control time-step)
	
	def __post_init__(self):
		self.total_window_size = self.input_width + self.io_offset
		
		self.input_slice = slice(0, self.input_width, self.input_step)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]
		
		self.output_start = self.total_window_size - self.output_width
		self.output_slice = slice(self.output_start, None, self.output_step)
		self.output_indices = np.arange(self.total_window_size)[self.output_slice]
		
		self.feature_columns = self.train_dfs[0].columns
		
		if self.output_columns is not None:
			self.output_col_indices = {col: i for i, col in enumerate(self.output_columns)}
		if self.input_columns is not None:
			self.input_col_indices = {col: i for i, col in enumerate(self.input_columns)}
		self.feature_col_indices = {col: i for i, col in enumerate(self.feature_columns)}
	
	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
			f'Input indices: {self.input_indices}',
			f'Output indices: {self.output_indices}',
			f'Input column name(s): {self.input_columns}'
			f'Output column name(s): {self.output_columns}'
		])
	
	def split_window(self, data_batches, multi=False):
		# data_batches.shape = (64, 120, 30), (batches, time-steps, features)
		inputs = data_batches[:, self.input_slice, :]
		
		if multi:
			# for all samples(0th dim of data_batches) and for all features (2nd dim of data_batches)
			# for each element i of self.input_slice (1st dim of outputs), get i + self.output_slice from data_batches 1st dim
			# and assign to 2nd dim of outputs
			# outputs = tf.zeros((tf.shape(data_batches)[0], int(self.input_width // self.input_step),
			#                     int(self.output_width // self.output_step), len(self.output_columns)))
			# for step_ahead in range(1, int(self.output_width // self.output_step) + 1):
			outputs = tf.expand_dims(data_batches[:, self.output_slice, :], axis=1)
			
			for step_ahead in range(self.input_slice.start + 1, self.input_slice.stop, self.input_slice.step):
				output_slice = slice(self.io_offset - self.output_width + step_ahead,
				                     self.io_offset + step_ahead,
				                     self.output_step)
				new_outputs = tf.expand_dims(data_batches[:, output_slice, :], axis=1)
				outputs = tf.concat([outputs, new_outputs], axis=1)
		else:
			outputs = data_batches[:, self.output_slice, :]
		
		if self.input_columns is not None:
			inputs = tf.stack(
				[inputs[:, :, self.feature_col_indices[col]] for col in self.input_columns],
				axis=-1,
			)
		
		if self.output_columns is not None:
			if multi:
				outputs = tf.stack(
					[outputs[:, :, :, self.feature_col_indices[col]] for col in self.output_columns],
					axis=-1
				)
			else:
				outputs = tf.stack(
					[outputs[:, :, self.feature_col_indices[col]] for col in self.output_columns],
					axis=-1
				)
		
		inputs.set_shape([None, int(self.input_width // self.input_step), None])
		
		if multi:
			outputs.set_shape([None, int(self.input_width // self.input_step), int(self.output_width // self.output_step), None])
		else:
			outputs.set_shape([None, int(self.output_width // self.output_step), None])
		
		return inputs, outputs
	
	# def merge_datasets(self, datasets):
	# 	pass
	
	def make_dataset(self, dfs, multi=False):
		
		for d, df in enumerate(dfs):
			data = np.array(df, dtype=np.float32)
			data = keras.utils.timeseries_dataset_from_array(
				data=data,
				targets=None,
				sequence_length=self.total_window_size,
				sequence_stride=1,
				shuffle=True,
				batch_size=64,
			)
			# ds = list(raw_data.as_numpy_iterator())
			# inputs = [inp for inp in ds]
			# # np.diff(inputs[0][0, :, 0])
			# for i in range(len(inputs)):
			# 	print(f'{(i / len(inputs)) * 100} %')
			# 	for j in range(inputs[i].shape[0]):
			# 		# print(f'{(j / outputs[i].shape[0]) * 100} %')
			# 		for k in range(inputs[i][j, :, :].shape[0]):
			# 			# print(f'{(j / outputs[i][j, :, :].shape[0]) * 100} %')
			# 			if np.isclose(inputs[i][j, k, 0], df['Time'].iloc[-1]):
			# 				print(i, j, k)
			# 				break
			
			# data.as_numpy_iterator().next().shape == (64, 120, 30), (batches, time-steps, features)
			if d == 0:
				dataset = data.map(lambda d: self.split_window(d, multi=multi))
			else:
				dataset = dataset.concatenate(data.map(lambda d: self.split_window(d, multi=multi)))
		
		# find most recent measurement
		#
		# ds = list(dataset.as_numpy_iterator())
		# inputs = [inp[0] for inp in ds]
		# outputs = [inp[1] for inp in ds]
		# inputs[-1][-1, :, 0]
		# # for i in range(len(inputs)):
		# # 	for j in range(inputs[i].shape[0]):
		# # 		for k in range(inputs[i][j, :, :].shape[0]):
		# for i in range(len(outputs)):
		# 	print(f'{(i / len(outputs)) * 100} %')
		# 	for j in range(outputs[i].shape[0]):
		# 		# print(f'{(j / outputs[i].shape[0]) * 100} %')
		# 		for k in range(outputs[i][j, :, :].shape[0]):
		# 			# print(f'{(j / outputs[i][j, :, :].shape[0]) * 100} %')
		# 			if np.isclose(outputs[i][j, k, :], df[self.output_columns].iloc[-1].to_numpy()).all():
		# 				print(i, j, k)
		# 				break
		
		return dataset
	
	def plot(self, X, y, model=None, model_name=None, plot_columns=None, max_subplots=5):
		cmap = plt.get_cmap('viridis')
		
		if plot_columns is None:
			plot_columns = self.output_columns
		
		norm = plt.Normalize(0, len(plot_columns))
		
		n_batches = len(X)
		fig, axs = plt.subplots(min(max_subplots, n_batches), 1, figsize=(48, 42), sharex=True, sharey=True)
		# print(plot_columns)
		for a, ax in enumerate(axs):
			ax.set(title=f'Batch {a + 1}')
			for c, col in enumerate(plot_columns):
				color = cmap(norm(c))
				
				plot_column_idx = self.input_col_indices[col]
				input_data = X[a, :, plot_column_idx] * (data_max[col] - data_min[col]) + data_min[col]
				ax.plot(self.input_indices, input_data,
				        marker='.', zorder=-10, label=f"T{col.split('_')[-1]} X",
				        color=color)
				
				output_column_idx = self.output_col_indices.get(col, None) if self.output_columns else plot_column_idx
				
				if output_column_idx is None:
					continue
				
				label_data = y[a, -1, :, output_column_idx] * (data_max[col] - data_min[col]) + data_min[col]
				ax.plot(self.output_indices, label_data,
				        linestyle=':', label=f"T{col.split('_')[-1]} y",
				        color=color)
				
				if model is not None:
					y_pred = model(X)
					if y_pred.shape[1] > len(self.output_indices):
						y_pred = np.reshape(y_pred,
						                    (y_pred.shape[0], len(self.input_indices), len(self.output_indices),
						                     len(self.output_columns)))[:, -1, :, :]
					
					pred_data = y_pred[a, :, output_column_idx] * (data_max[col] - data_min[col]) + data_min[col]
					ax.scatter(self.output_indices, pred_data,
					           marker='X', edgecolors='k', label=f"T{col.split('_')[-1]} yhat",
					           color=color, s=128)
					fig.savefig(f'./figs/{model_name}_pred_ts.png')
		
		# axs[0].set(title=f"{plot_columns[0].split('_')[0]}")
		axs[0].legend(bbox_to_anchor=(0.5, 2.5), loc='upper center', ncol=6)
		axs[-1].set(xlabel="Time [s]")
		axs[-1].set_xlim((0, int(axs[-1].get_xlim()[1])))
		axs[-1].set_xticks(range(0, int(axs[-1].get_xlim()[1]), 10))
		fig.tight_layout()
		fig.show()
		
		if model is not None:
			fig.savefig(f'./figs/{model_name}_pred_ts.png')
	
	@property
	def train(self):
		return self.make_dataset(self.train_dfs)
	
	@property
	def val(self):
		return self.make_dataset(self.val_dfs)
	
	@property
	def test(self):
		return self.make_dataset(self.test_dfs)
	
	@property
	def train_multi(self):
		return self.make_dataset(self.train_dfs, multi=True)
	
	@property
	def val_multi(self):
		return self.make_dataset(self.val_dfs, multi=True)
	
	@property
	def test_multi(self):
		return self.make_dataset(self.test_dfs, multi=True)

