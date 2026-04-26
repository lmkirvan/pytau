# This script demonstrates how to use FitHandler with customization
# Specifically, it shows how to overwrite spike-trains after loading
# and run the rest of the inference pipeline manually.
#
# NOTE: This approach is not advised for production code. It is preferred
# to follow the procedure in fit_manually.py for more control.
# However, this example is useful for understanding the FitHandler internals
# or when you need to make modifications to loaded data before fitting.

# Import modules
import os
import sys
from glob import glob

import pylab as plt

from pytau.changepoint_analysis import PklHandler
from pytau.changepoint_io import DatabaseHandler, FitHandler
from pytau.utils import plotting

try:
    pytau_base_dir = os.path.dirname(os.path.abspath(__file__))
except:
    pytau_base_dir = os.path.dirname(os.getcwd())
print(f"Using PyTAU base dir: {pytau_base_dir}")

# Find hf5 file
h5_path = glob(os.path.join(pytau_base_dir, "**", "*.h5"), recursive=True)[0]
data_dir = os.path.dirname(h5_path)

# Specify params for fit
model_parameters = dict(
    states=4,
    fit=40000,
    samples=20000,
    model_kwargs={"None": None},
)

preprocess_parameters = dict(
    time_lims=[2000, 4000],
    bin_width=50,
    data_transform="None",  # Can also be 'spike_shuffled','trial_shuffled'
)

FitHandler_kwargs = dict(
    data_dir=data_dir,
    taste_num=0,
    region_name="bla",  # Should match specification in info file
    laser_type=None,
    experiment_name="pytau_test",
)

# Initialize handler, and feed parameters
handler = FitHandler(**FitHandler_kwargs)
handler.set_model_params(**model_parameters)
handler.set_preprocess_params(**preprocess_parameters)

# ============================================================
# SECTION: HOW TO OVERWRITE SPIKE-TRAINS AFTER LOADING
# ============================================================
#
# This section demonstrates how to:
# 1. Load spike-trains using the handler's load_spike_trains method
# 2. Modify the loaded spike-trains (e.g., remove trials or neurons)
# 3. Run subsequent pipeline methods manually
#
# This is useful when you need to preprocess data before fitting
# but want to use the FitHandler infrastructure for the rest.

print("\n" + "=" * 60)
print("DEMO: Overwriting loaded spike-trains")
print("=" * 60)

# Step 1: Load spike trains using the handler
# This populates handler.data with the raw spike train array
handler.load_spike_trains()

# At this point, handler.data contains:
# - For taste_num (int): 3D array of shape (trials, neurons, time)
# - For taste_num="all": 4D array of shape (tastes, trials, neurons, time)

print(f"\nOriginal spike train shape: {handler.data.shape}")
print(f"Original number of trials: {handler.data.shape[0]}")
print(f"Original number of neurons: {handler.data.shape[1]}")

# Step 2: Modify the spike trains
# Example modifications:
# - Remove specific trials (e.g., trials with artifacts)
# - Remove specific neurons (e.g., poorly isolated units)
# - Perform custom preprocessing that FitHandler doesn't support

# Example: Remove first 2 trials and last 3 neurons
trials_to_keep = slice(2, None)  # Keep from trial 2 onwards
neurons_to_keep = slice(None, -3)  # Keep all neurons except last 3

modified_data = handler.data[trials_to_keep, neurons_to_keep, :]
print(f"\nModified spike train shape: {modified_data.shape}")
print(f"Modified number of trials: {modified_data.shape[0]}")
print(f"Modified number of neurons: {modified_data.shape[1]}")

# Alternative: Select specific trials/neurons
# specific_trials = [0, 2, 5]  # Keep trials 0, 2, and 5
# specific_neurons = [0, 1, 2, 5, 10]  # Keep neurons 0, 1, 2, 5, and 10
# modified_data = handler.data[specific_trials][:, specific_neurons, :]

# Step 3: Overwrite handler.data with modified data
# Now when we call preprocess_data(), it will use our modified data
handler.data = modified_data

# Step 4: Run subsequent pipeline methods manually/in a nested manner
# The methods check for existence of required attributes before running,
# so by having handler.data already set, preprocess_data() will skip loading

print("\nRunning preprocessing (using modified data)...")
handler.preprocess_data()

print("Creating model...")
handler.create_model()

print("Running inference...")
handler.run_inference()

# Save output to model database
print("Saving fit output...")
handler.save_fit_output()

print("\n" + "=" * 60)
print("Demo complete! FitHandler ran with modified spike-trains.")
print("=" * 60 + "\n")

# ============================================================
# END OF CUSTOMIZATION SECTION
# ============================================================

# Alternative: If you just want to run inference normally with FitHandler,
# you can skip the above customization section and just do:
# handler.run_inference()
# handler.save_fit_output()

# Access fit results
# Directly from handler
inference_outs = handler.inference_outs
# inference_outs contains following attributes
# model : Model structure
# approx : Fitted model
# lambda : Inferred firing rates for each state
# tau : Inferred changepoints
# data : Data used for inference

# Can also get path to pkl file from model database

fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

# Get fits for a particular experiment
dframe = fit_database.fit_database
wanted_exp_name = "pytau_test"
wanted_frame = dframe.loc[dframe["exp.exp_name"] == wanted_exp_name]
# Pull out a single data_directory
pkl_path = wanted_frame["exp.save_path"].iloc[0]

# Information saved in model database
# preprocess.time_lims
# preprocess.bin_width
# preprocess.data_transform
# preprocess.preprocessor_name
# model.states
# model.fit
# model.samples
# model.model_kwargs
# model.model_template_name
# model.inference_func_name
# data.data_dir
# data.basename
# data.animal_name
# data.session_date
# data.taste_num
# data.laser_type
# data.region_name
# exp.exp_name
# exp.model_id
# exp.save_path
# exp.fit_date
# module.pymc3_version
# module.theano_version

# From saved pkl file

this_handler = PklHandler(pkl_path)
# Can access following attributes
# Tau:
#   Raw Int tau : All tau samples in terms of indices of array given ==> this_handler.tau.raw_int_tau
#   Raw mode tau : Mode of samples in terms of indices of array given ==> this_handler.tau.raw_mode_tau
#   Scaled Tau : All tau samples scaled to stimulus delivery ==> this_handler.tau.scaled_tau
#   Int Scaled Tau : Integer values of "Scaled Tau" ==> this_handler.tau.scaled_int_tau
#   Mode Scale Tau : Mode of Int Scaled Tau ==> this_handler.tau.scaled_mode_tau
# Firing:
#   Raw spikes : Pulled using EphysData ==> this_handler.firing.raw_spikes
#   Mean firing rate per state : this_handler.firing.state_firing
#   Snippets around each transition : this_handler.firing.transition_snips
#   Significance of changes in state firing : this_handler.firing.anova_p_val_array
#   Significance of changes in firing across transitions : this_handler.firing.pairwise_p_val_array
# Metadata
this_handler.pretty_metadata

# Plotting
fit_model = this_handler.data["model_data"]["approx"]
spike_train = this_handler.firing.raw_spikes
scaled_mode_tau = this_handler.tau.scaled_mode_tau

# Plot ELBO over iterations, should be flat by the end
fig, ax = plotting.plot_elbo_history(fit_model)
plt.show()

# Overlay raster plot with states
fig, ax = plotting.plot_changepoint_raster(
    spike_train, scaled_mode_tau, [1500, 4000])
plt.show()

# Overview of changepoint positions
fig, ax = plotting.plot_changepoint_overview(scaled_mode_tau, [1500, 4000])
plt.show()

# Aligned spiking
fig, ax = plotting.plot_aligned_state_firing(spike_train, scaled_mode_tau, 300)
plt.show()

# Plot mean firing rates per state
fig, ax = plotting.plot_state_firing_rates(spike_train, scaled_mode_tau)
plt.show()
