from array import array
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import tensorflow as tf
import keras
from keras import layers

from helpers import generate_model
from init import *

def main():
	## LOAD BASE AND SIMULATION FLORIS WIND FARM MODELS
	
	# Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
	simulation_fi = generate_model(WAKE_FIELD_CONFIG["floris_input_file"])
	# model_fi = generate_model(BASE_MODEL_FLORIS_DIR)

	## GENERATE/LOAD TIME-SERIES OF FREE-STREAM WIND-SPEED, WIND DIRECTION, TURBINE AXIAL INDUCTION FACTORS AND YAW ANGLES
	# use array.array
	
	## BULD RNN/LSTM MODEL
	
	## START SIMULATION
	# GENERATE/LOAD TIME-SERIES OF ROTOR-EFFECTIVE WIND SPEEDS FOR BASE MODEL
	# TRAIN RNN/LSTM MODEL FOR X EPOCHS WITH TRAINING DATA COLLECTED UP TO NOW
    # MAKE PREDICTIONS FOR EACH TIME-STEP
    
    ## PLOT COST FUNCTION VS SIMULATION TIME-STEP