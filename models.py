from keras import layers
from keras.losses import Loss, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from keras.optimizers.legacy import Adam
from keras.layers import Input, GRU, LSTM, ConvLSTM2D, Dense, Reshape, LSTMCell, RNN, Lambda, Add, Layer, GRUCell, \
	TimeDistributed
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras import Model
import tensorflow as tf
import numpy as np
from constants import og_config, TUNABLE_MODEL_TYPES, MODEL_TYPE_OUTPUT_MAPPING, input_columns, \
	downstream_turbine_indices, yaw_angle_cols, yaw_angle_idx
from floris import tools as wfct

def numpy_params2pred(inputs, **kwargs):
	"""

	Args:
		inputs: vector of empirical gaussian parameters, bias term, freestreamWindVelX, freestreamWindVelY, yawAngles,

	Returns:

	"""
	params_input = inputs[0]
	training_input = inputs[1]
	
	config = dict(og_config)
	
	for model_type in TUNABLE_MODEL_TYPES:
		model_name = config['wake']['model_strings'][model_type]
		param_key = f'wake_{model_type.split("_")[0]}_parameters'
		list_idx = -1
		# MODEL_TYPE_OUTPUT_MAPPING[model_type]
		# is_list_param = False
		# idx = 0
		for param_name, idx in MODEL_TYPE_OUTPUT_MAPPING[model_type]:
			# for param_name in TUNABLE_MODEL_PARAMS[model_type]:
			if type(config['wake'][param_key][model_name][param_name]) is list:
				# if is_list_param:
				# 	# if the start of this list has already been parameterized
				# 	list_idx += 1
				list_idx += 1
				# assert np.isclose(config['wake'][param_key][model_name][param_name][list_idx], params_input[idx].numpy())
				# if hasattr(params_input[idx], 'numpy'):
				config['wake'][param_key][model_name][param_name][list_idx] = params_input[idx]
				if list_idx == len(config['wake'][param_key][model_name][param_name]) - 1:
					list_idx = -1
			# is_list_param = True
			else:
				# is_list_param = False
				# assert config['wake'][param_key][model_name][param_name] == params_input[idx].numpy()
				# if hasattr(params_input[idx], 'numpy'):
				config['wake'][param_key][model_name][param_name] = params_input[idx]
	# idx += 1
	
	# config['wake'][param_key][model_name].update(
	# 	{param_name: inputs[idx] for param_name, idx in zip(TUNABLE_MODEL_PARAMS[model_type],
	# 	                                                    MODEL_TYPE_OUTPUT_MAPPING[model_type])}
	# )
	
	fi = wfct.floris_interface.FlorisInterface(config)
	
	freestream_wind_vel = tf.stack([training_input[:, :, input_columns.index("FreestreamWindSpeedsX")],
	                                training_input[:, :, input_columns.index("FreestreamWindSpeedsY")]], axis=-1)
	
	freestream_wind_vel = (freestream_wind_vel * (kwargs["data_max"][["FreestreamWindSpeedsX", "FreestreamWindSpeedsY"]]
	                                              - kwargs["data_min"][["FreestreamWindSpeedsX", "FreestreamWindSpeedsY"]])) \
	                      + kwargs["data_min"][["FreestreamWindSpeedsX", "FreestreamWindSpeedsY"]]
	
	# freestream_wind_vel = np.zeros((training_input.shape[0], training_input.shape[1], 2))
	freestream_ws = tf.norm(freestream_wind_vel, axis=-1)
	# df[f'FreestreamWindSpeedsY'] = wind_speeds * -np.cos(wind_directions)
	# df[f'FreestreamWindSpeedsX'] = wind_speeds * -np.sin(wind_directions)
	freestream_wd = tf.math.asin(-(freestream_wind_vel[:, :, 0] / freestream_ws)) * (180 / np.pi)
	# y_preds = tf.zeros((freestream_ws.shape[0], freestream_ws.shape[1], len(output_columns)))
	y_preds = []
	for batch_idx in range(freestream_ws.shape[0]):
		# y_preds.append([])
		# for time_step_idx in range(freestream_ws.shape[1]):
		time_step_idx = -1
		fi.reinitialize(wind_speeds=[freestream_ws[batch_idx, time_step_idx]],
		                wind_directions=[freestream_wd[batch_idx, time_step_idx]])
		yaw_angles = (np.array([training_input[batch_idx, time_step_idx, i] for i in yaw_angle_idx]) \
		              * (kwargs["data_max"][yaw_angle_cols] - kwargs["data_min"][yaw_angle_cols])
		              + kwargs["data_min"][yaw_angle_cols]).to_numpy()
		fi.calculate_wake(yaw_angles=yaw_angles[np.newaxis, np.newaxis, :])
		# y_preds[batch_idx, time_step_idx, :] = \
		y_preds.append(fi.turbine_effective_velocities[0, 0, downstream_turbine_indices] + params_input[-1].numpy())
	return np.vstack(y_preds)


# except Exception as e:
# 	print(e)
# 	# return tf.zeros((training_input.shape[0], 1, len(data_obj.output_columns)))
# 	return tf.zeros((0, 1, len(data_obj.output_columns)))

@tf.function
def tf_params2pred(inputs):
	return tf.numpy_function(numpy_params2pred, inputs, tf.float32)


class Residual(Model):
	"""
	This model makes predictions for each time-step using the input from the previous time-step
	plus the delta calcualte by the model.
	"""
	
	def __init__(self, model, output_indices):
		super().__init__()
		self.model = model
		self.output_indices = output_indices
	
	def call(self, inputs, *args, **kwargs):
		delta = self.model(inputs, *args, **kwargs)
		return inputs[:, :, self.output_indices[0]:(self.output_indices[-1] + 1)]  + delta


class Feedback(Model):
	def __init__(self, units, out_steps, num_features, cell_class, output_indices):
		super().__init__()
		self.out_steps = out_steps
		self.output_indices = output_indices
		self.units = units
		self.cell = cell_class(units)
		self.rnn = RNN(self.cell, return_state=True)
		self.dense = Dense(num_features) # neuron for every feature
		self.reshape = Lambda(lambda x: x[:, self.output_indices[0]:(self.output_indices[-1] + 1)]) # just select output features
	
	def warmup(self, inputs):
		x, *state = self.rnn(inputs)
		y_pred = self.dense(x)
		# print(142, y_pred, state, sep='\n')
		return y_pred, state
	
	def call(self, inputs, training=None):
		# tensor-array to capture dynamically unrolled outputs
		y_preds = []
		
		# initialize the LSTM state
		y_pred, state = self.warmup(inputs)
		
		# insert first prediction
		y_preds.append(self.reshape(y_pred))
		# print(154, self.reshape(y_pred), sep='\n')
		
		# run the rest of the prediction steps
		for i in range(1, self.out_steps):
			# use the last prediction as input, should contain all features
			x = y_pred
			# print(160, x, sep='\n')
			
			# execute a single lstm step
			x, state = self.cell(x, states=state, training=training)
			
			# print(165, x, state, sep='\n')
			
			# convert the lstm output to a prediction
			y_pred = self.dense(x)
			
			# print(170, y_pred, sep='\n')
			# print(171, self.reshape(y_pred), sep='\n')
			
			# add the prediction to the output
			y_preds.append(self.reshape(y_pred))
		
		y_preds = tf.stack(y_preds)
		y_preds = tf.transpose(y_preds, [1, 0, 2])
		return y_preds


class ResidualFeedback(Model):
	def __init__(self, units, out_steps, num_features, cell_class, output_indices):
		super().__init__()
		self.out_steps = out_steps
		self.output_indices = output_indices
		self.units = units
		self.cell = cell_class(units)
		self.rnn = RNN(self.cell, return_state=True)
		self.dense = Dense(num_features, kernel_initializer=tf.initializers.zeros())
		self.reshape = Lambda(lambda x: x[:, self.output_indices[0]:(self.output_indices[-1] + 1)])
	
	def warmup(self, inputs):
		# inputs shape=(batch=None, time-steps=60, features=20)
		x, *state = self.rnn(inputs)
		delta = self.dense(x)
		
		# delta = Tensor("feedback/dense/BiasAdd:0", shape=(None, 20), dtype=float32)
		# state = [<tf.Tensor 'feedback/rnn/while:4' shape=(None, 32) dtype=float32>, <tf.Tensor 'feedback/rnn/while:5' shape=(None, 32) dtype=float32>]
		# adding delta to last-time-step of each batch
		y_pred = inputs[:, -1, :] + delta
		
		return y_pred, state
	
	def call(self, inputs, training=None):
		# array to capture dynamically unrolled outputs
		y_preds = []
		
		# initialize the LSTM state
		y_pred, state = self.warmup(inputs)
		# insert first prediction
		# self.reshape_2(y_pred) = Tensor("feedback/lambda/strided_slice_1:0", shape=(None, 6), dtype=float32)
		
		y_preds.append(self.reshape(y_pred))
		
		# run the rest of the prediction steps
		for i in range(1, self.out_steps):
			# use the last prediction as input, should contain all features
			x = y_pred
			
			# x = Tensor("feedback/dense/BiasAdd:0", shape=(None, 20), dtype=float32)
			
			# execute a single lstm step
			x, state = self.cell(x, states=state, training=training)
			
			# x = Tensor("feedback/lstm_cell/mul_2:0", shape=(None, 32), dtype=float32)
			# state = [<tf.Tensor 'feedback/lstm_cell/mul_2:0' shape=(None, 32) dtype=float32>, <tf.Tensor 'feedback/lstm_cell/add_1:0' shape=(None, 32) dtype=float32>]
			
			# convert the lstm output to a prediction
			# delta = self.reshape_1(self.dense(x))
			delta = self.dense(x)
			
			# delta = Tensor("feedback/dense/BiasAdd_1:0", shape=(None, 20), dtype=float32)
			y_pred = y_pred + delta
			
			# add the prediction to the output
			# self.reshape_2(y_pred) = Tensor("feedback/lambda/strided_slice_2:0", shape=(None, 6), dtype=float32)
			y_preds.append(self.reshape(y_pred))
		
		y_preds = tf.stack(y_preds)
		y_preds = tf.transpose(y_preds, [1, 0, 2])
		return y_preds


