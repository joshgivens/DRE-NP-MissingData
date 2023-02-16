
All code is in Python. Requires use of the `Torch`, `numpy` and `scikitlearn` libraries.
# File Structure

* `functions`: The `functions` folder contains all of the functions required to implement our procedures. 

* `plots`: Contains all the summary plots from the paper.

* `results`: Contains the results from the simulated experiments in subfolder `simulated_results` and real world experiments in subfolder `real_world_results`.

* `real_world_data`: The real world data used in our experiments.

* `simulation_and_plot_code`: Contains code and notebooks to run all our simulated and real world experiments as well as plot the results.

# Detail on files in `functions`
We now briefly describe these files:    
* `objective_funcs_torch.py` contains the objective function that is minimised in KLIEP and M-KLIEP and any other necessary function.

* `gradient_descent_torch.py` contains a gradient descent algorithm to optimise the objective functions

* `estimators_torch.py` contains function which wrap the gradient descent with the objective to perform KLIEP, M-KLIEP, etc.

* `data_sim_framework_torch.py` contains a function to repeat multiple iterations of simulated experiments from generating data to performing DRE technique.

* `np_classifier_torch.py` all the functions that perform np classification given a score function.

* `pipeline_funcs.py` functions perform the full procedure for our real world experiments. 

# Detail on files in `simulation_and_plot_code`


* `datagen_kliep_foraistat.py` Contains code to run all simulated experiments and save results to `results/real_world_results`.
* `plot_kliep_comparison_foraistat.ipynb` Contains code to plot results from simulated experiments and saves them to `plots`.

* `CTG_dre.ipynb` Performs real-world experiments and plots results for the CTG data found in `real_world_data/CTG.xls`.
* `Smoke_detection.ipynb` Performs real-world experiments and plots results for the Fire data found in `real_world_data/smoke_detection_iot.csv`.
* `WeatherAus.ipynb` Performs real-world experiments and plots results for the Weather data found in `real_world_data/weatherAUS.csv`.