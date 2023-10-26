import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import keras
from keras import layers

DATA_DIR = os.path.join('/Users', 'aoifework', 'Documents', 'Research', 'wake_modeling', 'SOWFA data', 'Output_SOWFA_TotalControl')
SIMULATION_TYPES = ['waked2a', 'waked2a.pcontrol', 'waked2a.ptcontrol', 'waked2a.ptlcontrol',
                 'waked4a', 'waked4a.pcontrol', 'waked4a.ptcontrol', 'waked4a.ptlcontrol']
# INPUT_COLS = ['torqueGen', 'nacYaw', 'pitch', 'Vaxial', 'Vmag', 'Vradial', 'Vtangential']
INPUT_COLS = ['torqueGen', 'pitch', 'Vaxial']
LOAD_DATA = False
CUT_TRANSIENTS = 200

if __name__ == '__main__':
    ## READ DATA
    if LOAD_DATA:
        # PC = wake-loss compensator, TB - thrust balancer, LL = Load-limiting controller
        base_cols = ['#Turbine', 'Sector', 'Time(s)', 'dt(s)']
        # data_df = pd.DataFrame(index_cols=['Time(s)', '#Turbine'], columns=['Sector', 'dt(s)'])
        data_df = pd.DataFrame(columns=base_cols)
        # data_df.set_index(['Time(s)', '#Turbine'], inplace=True)
        metric_types = set()
        control_types = set()
        for root, dirs, filenames in os.walk(DATA_DIR):
            valid_dir = False
            valid_ctrl_type = None
            valid_wake_scenario = None
            for sim_type in SIMULATION_TYPES:
                dirname = os.path.basename(os.path.abspath(os.path.join(root, '..', '..')))
                ctrl_type_idx = dirname.find('waked')
                if sim_type == dirname[ctrl_type_idx:]:
                    valid_dir = True
                    
                    valid_wake_scenario = sim_type if '.' not in sim_type else sim_type.split('.')[0]
                    valid_ctrl_type = 'nocontrol' if '.' not in sim_type else sim_type.split('.')[-1]
                    
                    control_types = control_types | {valid_ctrl_type}
                    break
            if valid_dir:
                for fn in filenames:
                    if fn not in INPUT_COLS:
                        continue
                        
                    # ctrl_type = os.path.basename(os.path.abspath(os.path.join(root, '..', '..'))).split('.')[-1]
                    with open(os.path.join(root, fn)) as f:
                        headers = f.readline()
                    headers = headers.split('    ')
                    headers[-1] = headers[-1].replace('\n', '') # remove space
                    metric_type = headers[-1]
                    metric_types = metric_types | {metric_type}
                    
                    tmp_df = pd.read_csv(os.path.join(root, fn), delimiter=' ', index_col=False, skiprows=lambda x: x != 1)
                    n_cols = len([col for col in tmp_df.columns if 'Unnamed' not in col])
                    
                    if n_cols - len(headers) > 0:
                        headers = headers[:-1] + [headers[-1] + str(i+1) for i in range(n_cols - len(headers) + 1)]
                    
                    tmp_df = pd.read_csv(os.path.join(root, fn), delimiter=' ', index_col=False, names=headers, header=0)
                    if 'Sector' in headers:
                        tmp_df.drop(columns=['Sector', 'dt(s)'], inplace=True)
                    else:
                        tmp_df.drop(columns=['dt(s)'], inplace=True)
                    
                    if 'Vaxial' in headers[-1]:
                        tmp_df[f'RotorAverageV'] = tmp_df.loc[:, [col for col in tmp_df.columns if
                                                                            'Vaxial' in col]].to_numpy().mean(axis=1)
                    
                        tmp_df.drop(columns=[col for col in tmp_df.columns if 'Vaxial' in col], inplace=True)
                        metric_type = 'RotorAverageV'
                        
                    tmp_df = tmp_df[tmp_df['Time(s)'] > tmp_df['Time(s)'].min() + CUT_TRANSIENTS]
                    tmp_df['Time(s)'] = tmp_df['Time(s)'] - tmp_df['Time(s)'].min()
                    
                    tmp_df['WakeScenario'] = [valid_wake_scenario] * len(tmp_df.index)
                    tmp_df['ControlType'] = [valid_ctrl_type] * len(tmp_df.index)
                    # tmp_df.rename(columns={col: col + '_' + valid_ctrl_type for col in tmp_df.columns[2:]}, inplace=True)
                    
                    if len(data_df.index) == 0:
                        data_df = tmp_df
                    else:
                        if ((data_df['ControlType'] == valid_ctrl_type) & (data_df['WakeScenario'] == valid_wake_scenario)).any():
                            if metric_type in data_df.columns:
                                data_df.loc[(data_df['ControlType'] == valid_ctrl_type) & (data_df['WakeScenario'] == valid_wake_scenario), metric_type] = tmp_df[metric_type]
                            else:
                                data_df = pd.merge(data_df, tmp_df,
                                                   on=['#Turbine', 'Time(s)', 'WakeScenario', 'ControlType'])
                        else:
                            data_df = pd.concat([data_df, tmp_df])
                    
        
        data_df.to_pickle(os.path.join(DATA_DIR, 'sowfa_data'))
        
        with open(os.path.join(DATA_DIR, 'control_types'), 'wb') as f:
            pickle.dump(control_types, f)
        
        with open(os.path.join(DATA_DIR, 'metric_types'), 'wb') as f:
            pickle.dump(metric_types, f)
        
        
    else:
        data_df = pd.read_pickle(os.path.join(DATA_DIR, 'sowfa_data'))
        
        with open(os.path.join(DATA_DIR, 'control_types'), 'rb') as f:
            control_types = pickle.load(f)
        
        with open(os.path.join(DATA_DIR, 'metric_types'), 'rb') as f:
            metric_types = pickle.load(f)
    
    # time = sorted(pd.unique(data_df['Time(s)']))
    if False:
        turbine_indices = pd.unique(data_df['#Turbine'])
        wake_scenarios = pd.unique(data_df['WakeScenario'])
        control_types = pd.unique(data_df['ControlType'])
        for wake_scenario in wake_scenarios:
            wake_scenario_df =  data_df.loc[data_df['WakeScenario'] == wake_scenario, :]
            for ctrl_type in control_types:
                ctrl_type_df = wake_scenario_df.loc[wake_scenario_df['ControlType'] == ctrl_type, :]
                fig, axs = plt.subplots(len(ctrl_type_df.columns) - 4, 1, sharex=True)  # subtract time and turbine index and , 'WakeScenario'
                for c, col in enumerate([col for col in ctrl_type_df.columns if col not in ['#Turbine', 'Time(s)', 'WakeScenario', 'ControlType']]):
                    for turbine_idx in turbine_indices:
                        turbine_df = ctrl_type_df.loc[ctrl_type_df['#Turbine'] == turbine_idx, :]
                        axs[c].plot(turbine_df['Time(s)'], turbine_df[col])
                        axs[c].set(title=col)
                axs[-1].set(xlabel='Time [s]')
                fig.show()
    
    # labels are vectors of turbine RotorAverageV at each time step
    # training_df = data_df.loc[(data_df['WakeScenario'] == 'waked2a') & (data_df['ControlType'] == 'nocontrol')]\
    #     .drop(columns=['WakeScenario', 'ControlType'])\
    #     .pivot(columns=['#Turbine'], index='Time(s)')
    training_df = data_df.pivot(columns=['#Turbine'], index=['Time(s)', 'WakeScenario', 'ControlType'])
    
    # training_df['WakeScenario'] = [training_df.index[i][training_df.index.names.index('WakeScenario')]
    #                                for i in range(len(training_df.index))]
    
    for col in ['WakeScenario', 'ControlType']:
        for i, val in enumerate(training_df.index.levels[training_df.index.names.index(col)]):
            # training_df.loc[training_df['WakeScenario'] == val, 'WakeScenario'] = i
            training_df[f'{col}_{val}'] = [1 if training_df.index[j][training_df.index.names.index(col)] == val else 0
                                                  for j in range(len(training_df.index))]
        
    training_df.reset_index(level=['WakeScenario', 'ControlType'], drop=True, inplace=True)
    
    # Allocate Training/Test/Validation Data
    training_end_idx = int(0.9 * len(training_df.index))
    test_end_idx = int(0.95 * len(training_df.index))
    y = training_df['RotorAverageV']
    X = training_df[['blade pitch angle (degrees)', 'generator torque (N-m)']]
    y_train = y.iloc[:training_end_idx]
    X_train = X.iloc[:training_end_idx]
    y_test = y.iloc[training_end_idx:test_end_idx]
    X_test = X.iloc[training_end_idx:test_end_idx]
    y_val = y.iloc[test_end_idx:]
    X_val = X.iloc[test_end_idx:]
    
    # Normalize Data
    y_rng = (y.min(), y.max())
    y_train = (y_train - y_rng[0]) / (y_rng[1] - y_rng[0])
    y_test = (y_test - y_rng[0]) / (y_rng[1] - y_rng[0])
    y_val = (y_val - y_rng[0]) / (y_rng[1] - y_rng[0])
    
    X_rng = (X.min(), X.max())
    X_train = (X_train - X_rng[0]) / (X_rng[1] - X_rng[0])
    X_test = (X_test - X_rng[0]) / (X_rng[1] - X_rng[0])
    X_val = (X_val - X_rng[0]) / (X_rng[1] - X_rng[0])
    
    # Make model
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=len(X_train.columns), output_dim=128))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(len(y_train.columns)))
    model.summary()
    
    # Training configuration
    model.compile(
        optimizer=keras.optimizers.legacy.RMSprop(), # Optimizer
        loss=keras.losses.MeanSquaredError(), # Loss function
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    print("Fit model on training data")
    history = model.fit(
        X_train,
        y_train,
        batch_size=100, # int(0.05 * len(X_train.index)),
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_val, y_val),
    )
    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    y_pred = model.predict(X_test.iloc[:3])
    y_pred = y_pred * (y_rng[1] - y_rng[0]).to_numpy() + y_rng[0].to_numpy()
    print("predictions shape:", y_pred.shape)
    y_true = y_test.iloc[:3] * (y_rng[1] - y_rng[0]) + y_rng[0]
    
    fig, axs = plt.subplots(1, 1)
    axs.scatter(range(y_pred.shape[1]), y_pred[0, :], marker='o')
    axs.scatter(y_true.columns.to_numpy(), y_true.iloc[0].to_numpy(), marker='*')
    fig.show()