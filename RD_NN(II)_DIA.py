# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:54:00 2023

@author: Andy
Thesis Rut-Depth prediction Neural Network Deep Learning model,
path(II).

# SHPA included.
Hyper Parameters Tuning, HPT, find the best combination.
"""

# Trilayer Neural Network model with hyperparameters of : auto execute
# with Heatmap using SHAP... not yet confirmed
# to be continued...

# Import and Preprocessing Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

total_start_time = time.time()

# Define tuning variables(hyperparameters):
#====================
custom_alpha=0.0001 # default=0.0001
LR = float(0.00001) # default=0.001, 'constant'
# Define the list of random states
random_states=0#0, 42, 50, 100, 200, 250, 500]
max_iteration=100000000 # 5000000
#====================

# # Define the Allowance range of ±1.5
allowY = (1.5)
#allowY = np.array(allowY)

# Create a DataFrame to store the outputs
output_df = pd.DataFrame(columns=['Output', 'Value', 'Layers','Activation','Solver'])

inputTable = pd.read_excel('datatable_v4_retain.xlsx', engine='openpyxl',sheet_name='retain(3)', header=0)

colname = ["Virgin Agg.(%)","RAP(%)",'1 1/2"','1"','3/4"','1/2"','3/8"',"#4","#8","#16","#30","#50","#100","#200","NMAS",\
           'Binder Type','Va(%)',"Binder Content(%)",'Testing Temperature']
    
# # Min-Max Scaling (最小-最大縮放)
# normalized_data = (inputTable - np.min(inputTable)) / (np.max(inputTable) - np.min(inputTable))

# # Z-score Normalization (或稱為Standardization，標準化)
# standardized_data = (inputTable - np.mean(inputTable)) / np.std(inputTable)

# # Min-Max Scaling
# scaler = MinMaxScaler()
# data_rescaled = scaler.fit_transform(inputTable.reshape(-1, 1))

# # Z-score Normalization
# scaler = StandardScaler()
# data_standardized = scaler.fit_transform(inputTable.reshape(-1, 1))

# 當進行正規化時，要注意以下幾點：
# 若訓練集已被正規化，那麼在進行預測或評估時，必須使用相同的轉換參數（如最小、最大值或平均值、標準差）對測試集或新資料進行正規化。
# 不是所有的算法都需要資料正規化。例如，決策樹或隨機森林通常不受尺度影響，而像是K-近鄰、支援向量機或神經網路這類的算法則可能會受到尺度的影響。
    
# define predictors and response
X_train = inputTable[colname].values
y_train = inputTable['RD(mm)'].values
y_train = np.where(y_train > 12.5, 12.5, y_train)
    
# Load the test data from the "Test" sheet
test_table = pd.read_excel('datatable_v4_retain.xlsx', engine='openpyxl', sheet_name='Test2', header=0)

# define test predictors and response
X_test = test_table[colname].values
y_test = test_table['RD(mm)'].values
y_test = np.where(y_test > 12.5, 12.5, y_test)

# # Min-Max Scaling
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# # y_train = scaler.fit_transform(y_train)
# y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
# X_test = scaler.transform(X_test)  # Use the same scaler fit from the training data
# # y_test = scaler.transform(y_test)
# y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

# # Scalers for features and target
# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# # Min-Max Scaling for features
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)

# # Min-Max Scaling for target
# y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Initialize variables to track the best parameters and scores
best_mse = np.inf
best_params_mse = None
best_r2 = -np.inf
best_params_r2 = None
best_pass = -np.inf
best_params_pass_layers = None
best_params_pass_actifunc = None
best_params_pass_solver = None
best_passrate = -np.inf
best_params_passrate_layers = None
best_params_passrate_actifunc = None
best_params_passrate_solver = None
# best_rmse = np.inf
# best_params_rmse_layers = None
# best_params_rmse_actifunc = None
# best_params_rmse_solver = None
# best_mae = np.inf
# best_params_mae_layers = None
# best_params_mae_actifunc = None
# best_params_mae_solver = None
# best_L2 = np.inf
# best_params_L2_layers = None
# best_params_L2_actifunc = None
# best_params_L2_solver = None

hidden_layer_sizes = [(i) for i in range(10, 110, 10)]
activation = ['relu','tanh','logistic']
solver = ['adam','lbfgs']

matplotlib.rcParams.update({'font.size': 14})

# Grid Search
for layers in hidden_layer_sizes:
    for acti_func in activation:
        for custom_solver in solver:

#for rnd_st8 in random_states:
            start_time = time.time()
            print('///////////////////////////////////////')
            #print(f'Running with random state = {rnd_st8}')
            
            # Output code information:
            print('======= ANN ML Bio =======')
            print('layers =',layers)
            print('activation function =', acti_func)
            print('solver =', custom_solver)
            print('alpha =', custom_alpha)
            print("LR =", LR)
            print('random state =',random_states)
            print('==========================')
            print('--------------------------')
            
            # Output code information:
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'layers', 'Value': layers}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'activation function', 'Value': acti_func}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'solver', 'Value': custom_solver}])], ignore_index=True)
            
            regressionNeuralNetwork = MLPRegressor(
                hidden_layer_sizes=layers,
                activation=acti_func,
                alpha=custom_alpha,
                max_iter=max_iteration,
                solver=custom_solver,
                random_state=random_states,
                
                # hyperparameters for future tuning but set as default for the time being.
                learning_rate_init = LR, #float, default=0.001
                n_iter_no_change = 10, #int, default=10
                max_fun = 5000000 # int, default=15000 # Only used when solver='lbfgs'
            )
        
      
            regressionNeuralNetwork.fit(X_train, y_train)
        
            # Predict on the test data
            y_pred = regressionNeuralNetwork.predict(X_test)
            
            # Set values above 12.5 to be 12.5
            y_pred = np.where(y_pred > 12.5, 12.5, y_pred)
            
            validationPredictions = regressionNeuralNetwork.predict(X_train)
            validationPredictions = np.where(validationPredictions > 12.5, 12.5, validationPredictions)
                    
            # Output results for the train-test split:
            print('======== results:=========')
            print("Iterations = ", regressionNeuralNetwork.n_iter_)
            mse = mean_squared_error(y_test, y_pred)
            print("MSE = {:.3f}".format(round(mse, 3)))
            r2 = r2_score(y_test, y_pred)
            print("R2 = {:.3f}".format(round(r2, 3)))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print("RMSE = {:.3f}".format(round(rmse, 3)))
            mae = mean_absolute_error(y_test, y_pred)
            print("MAE = {:.3f}".format(round(mae, 3)))
            L2 = np.linalg.norm(y_test - y_pred) / np.linalg.norm(y_test)
            print("L2 = {:.3f}".format(round(L2, 3)))
            ### Training time
            end_time = time.time()
            training_time = end_time - start_time
            print("Training time: {:.3f} seconds".format(round(training_time, 3)))
            print('==========================')
            print('--------------------------')
            
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'epochs', 'Value': regressionNeuralNetwork.n_iter_}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'MSE', 'Value': round(mse,3)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'R2', 'Value': round(r2,3)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'RMSE', 'Value': round(rmse,3)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'MAE', 'Value': round(mae,3)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'L2', 'Value': round(L2,3)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Training Time', 'Value': round(training_time,3)}])], ignore_index=True)
            
            # Extract true response values
            trueResponse = np.where(y_test > 12.5, 12.5, y_test)
            predictedResponse = y_pred
            
            # Create a vector of record numbers
            recordNumber = np.arange(len(trueResponse)) + 1
        
            # Convert y_test and recordNumber to arrays
            y_test = np.array(y_test)
            recordNumber = np.array(recordNumber)
            
            ### Plot the Scatter plot with allowY range
            plt.figure(figsize=(12,8))
            plt.plot(recordNumber, trueResponse, '.', color=[0, 0.41, 0.59], markersize=7)
            plt.plot(recordNumber, predictedResponse, '*', color=[1, 0.5, 0], markersize=5)
            # Connect the lines to the corresponding data points
            plt.xlabel('Record Number(#)')
            plt.ylabel('RutDepth(mm)')
            # Plot the straight lines
            for x, y in zip(recordNumber, trueResponse):
                plt.plot([x, x], [y + allowY, y - allowY], '--', color=[0.7, 0.7, 0.7])
            plt.title(f'Scatter Plot of neurons:{layers} σ:{acti_func} Solver:"{custom_solver}" α:{custom_alpha} LR:{LR} rs:{random_states}')
            plt.legend(['True Response', 'Predicted Response', f'Allowance:{allowY}'], loc='best')
            plt.show()

            #=============
            # Plot actual vs predicted values
            plt.figure(figsize=(12, 12))
            plt.scatter(y_test, y_pred, alpha=0.5, color='b')
            # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='r', lw=1.5)
            plt.plot([0, y_test.max()+5], [0, y_test.max()+5], '--', color='r', lw=1.5)
            plt.xlabel('Actual(mm)')
            plt.ylabel('Predicted(mm)')
            plt.title(f'Actual vs. Predicted neurons:{layers} σ:{acti_func} Solver:"{custom_solver}" α:{custom_alpha} LR:{LR} rs:{random_states} R-square={r2:.3f}')
            plt.axis([0, y_test.max()+5, 0, y_test.max()+5])
            #plt.grid(True)
            plt.legend(['Data','Ideal Line:y=x'], loc='best')
            plt.show()
            #-------------
            
            # Plot Residual
            residuals = trueResponse - predictedResponse
            residuals = trueResponse - predictedResponse
            plt.figure(figsize=(12,8))
            plt.plot(recordNumber, residuals, '.', color=[1, 0.3, 0.1], markersize = 7)
            plt.axhline(y=0, color=[0.2, 0.6, 0.2], linestyle='dashed')
            plt.axhline(y=allowY, color=[0.7, 0.7, 0.7], linestyle='dashed')
            plt.axhline(y=-allowY, color=[0.7, 0.7, 0.7], linestyle='dashed')
            plt.xlabel('Data(#)')
            plt.ylabel('Residuals(mm)')
            plt.title(f'Residual Plot of neurons:{layers} σ:{acti_func} Solver:"{custom_solver}" α:{custom_alpha} LR:{LR} rs:{random_states}')
            # plt.ylim([-20, 20])  # Set y-axis range
            plt.legend(['Predicted Response','Reference', f'Allowance:{allowY}'], loc='best')
            plt.show()
            
            # Plot loss function
            if custom_solver == 'adam' or custom_solver == 'sgd':
                plt.figure(figsize=(12,8))
                plt.plot(regressionNeuralNetwork.loss_curve_)
                plt.xlabel('Iteration')
                plt.ylabel('MSE')
                plt.title(f'Loss Curve {layers} σ:{acti_func} Solver:"{custom_solver}" α:{custom_alpha} LR:{LR} rs:{random_states}')
                plt.legend(['Loss Curve'],loc='best')
                plt.show()
            else:
                print('--------------------------')
                print("'loss_curve_' Only accessible when solver='sgd'or'adam'.")
                
            # =====================================
            # Plot Count: Identify pass or fail based on allowance range
            # =====================================
            
            # Check if residuals are within allowance
            pass_fail = np.where((residuals <= allowY) & (residuals >= -allowY), "Pass", "Failed")
            unique, counts = np.unique(pass_fail, return_counts=True)
            pass_fail_dict = dict(zip(unique, counts))
            
            # Plot counts
            plt.figure(figsize=(12,8))
            # Compute percentage
            total = sum(pass_fail_dict.values())
            pass_percentage = pass_fail_dict.get('Pass', 0) / total * 100
            bars = plt.bar(['Pass', 'Fail'], [pass_fail_dict.get('Pass', 0), pass_fail_dict.get('Failed', 0)], color=['green','red'])
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.title(f'Distribution of Pass and Failed Predictions of neurons:{layers} σ:{acti_func} Solver:"{custom_solver}" α:{custom_alpha} LR:{LR} rs:{random_states}')
            plt.ylim([0, total+1])  # Set y-axis range
            # Create labels for the legend
            # Check if there's more than one element in `counts`
            pass_count = pass_fail_dict.get('Pass', 0)
            fail_count = pass_fail_dict.get('Failed', 0)
            labels = [f"Pass:{pass_count}({pass_percentage:.2f}%)", f"Fail:{fail_count}({100-pass_percentage:.2f}%)"]
            plt.legend([bars[0], bars[1]], [labels[0],labels[1]])
            plt.show()       
            
            # =====================================
            # Pass-fail results
            # =====================================
            
            print(f'Allowance = {allowY}')
            print('========Pass-Fail=========')
            print(f"Pass:{pass_count}({pass_percentage:.2f}%)")
            print(f"Fail:{fail_count}({100-pass_percentage:.2f}%)")
            print('==========================')
            print('///////////////////////////////////////')
            print('')
            
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Pass Count', 'Value': pass_count}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Fail Count', 'Value': fail_count}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Pass Percentage', 'Value': round(pass_percentage,2)}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Fail Percentage', 'Value': round(100-pass_percentage,2)}])], ignore_index=True)
                        
        
            # Update the best parameters and the smallest MSE
            if mse < best_mse:
                best_mse = mse
                best_params_mse_layers = (layers)
                best_params_mse_actifunc = (acti_func)
                best_params_mse_solver = (custom_solver)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params_r2_layers = (layers)
                best_params_r2_actifunc = (acti_func)
                best_params_r2_solver = (custom_solver)
                
            if pass_count > best_pass:
                best_pass = pass_count
                best_params_pass_layers = (layers)
                best_params_pass_actifunc = (acti_func)
                best_params_pass_solver = (custom_solver)
                
            if pass_percentage > best_passrate:
                best_passrate = pass_percentage
                best_params_passrate_layers = (layers)
                best_params_passrate_actifunc = (acti_func)
                best_params_passrate_solver = (custom_solver)
            
            # if rmse > best_rmse:
            #     best_rmse = rmse
            #     best_params_rmse_layers = (layers)
            #     best_params_rmse_actifunc = (acti_func)
            #     best_params_rmse_solver = (custom_solver)
                
            # if mae > best_mae:
            #     best_mae = mae
            #     best_params_mae_layers = (layers)
            #     best_params_mae_actifunc = (acti_func)
            #     best_params_mae_solver = (custom_solver)
                
            # if L2 > best_L2:
            #     best_L2 = L2
            #     best_params_L2_layers = (layers)
            #     best_params_L2_actifunc = (acti_func)
            #     best_params_L2_solver = (custom_solver)
                
            # Print the best parameters and the corresponding MSE
            print("So far Best parameters by MSE: ", best_params_mse_layers,',',best_params_mse_actifunc,',',best_params_mse_solver)
            print(f"Best MSE: {round(best_mse, 3):.3f}")
        
            # Print the best parameters and the corresponding R2
            print("So far Best parameters by R2: ", best_params_r2_layers,',',best_params_r2_actifunc,',',best_params_r2_solver)
            print(f"Best R2: {round(best_r2, 3):.3f}")
            
            # Print the best parameters by the corresponding Pass count
            print("So far Best parameters by Pass count: ", best_params_pass_layers,',',best_params_pass_actifunc,',',best_params_pass_solver)
            print(f"Best Pass count: {best_pass:}")
            
            # Print the best parameters by the corresponding Pass rate
            print("So far Best parameters by Pass rate: ", best_params_passrate_layers,',',best_params_passrate_actifunc,',',best_params_passrate_solver)
            print(f"Best Pass rate: {round(best_passrate, 2):.2f}")
            
            print('')
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': "-", 'Value': "", 'Layers': "", 'Activation': "", 'Solver':""}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good MSE', 'Value': round(best_mse,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good R2', 'Value': round(best_r2,3), 'Layers': best_params_r2_layers, 'Activation': best_params_r2_actifunc, 'Solver':best_params_r2_solver}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good Pass count', 'Value': best_pass, 'Layers': best_params_pass_layers, 'Activation': best_params_pass_actifunc, 'Solver':best_params_pass_solver}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good Passrate', 'Value': round(best_passrate,2), 'Layers': best_params_passrate_layers, 'Activation': best_params_passrate_actifunc, 'Solver':best_params_passrate_solver}])], ignore_index=True)
            # output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good RMSE', 'Value': round(best_rmse,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
            # output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good MAE', 'Value': round(best_mae,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
            # output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Good L2', 'Value': round(best_L2,3), 'Layers': best_params_r2_layers, 'Activation': best_params_r2_actifunc, 'Solver':best_params_r2_solver}])], ignore_index=True)
            output_df = pd.concat([output_df, pd.DataFrame([{'Output': "--", 'Value': "", 'Layers': "", 'Activation': "", 'Solver':""}])], ignore_index=True)
            
            # # Create a model agnostic Kernel SHAP explainer
            explainer = shap.KernelExplainer(regressionNeuralNetwork.predict, X_train)
                
            # # Calculate SHAP values
            shap_values = explainer.shap_values(X_train, nsamples=100)

            # # Plot the SHAP values
            shap.summary_plot(shap_values, X_train, feature_names=colname)
            shap.summary_plot(shap_values, X_train, plot_type='bar', color='m', feature_names=colname)
            
# sys.stdout = original_stdout
# f.close()
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
print("Total Training time: {:.3f} seconds\n".format(round(total_training_time, 3)))

# Print the best parameters and the corresponding MSE
print("Best parameters by MSE: ", best_params_mse_layers,',',best_params_mse_actifunc,',',best_params_mse_solver)
print(f"Best MSE: {round(best_mse, 3):.3f}")

# Print the best parameters and the corresponding MSE
print("Best parameters by R2: ", best_params_r2_layers,',',best_params_r2_actifunc,',',best_params_r2_solver)
print(f"Best R2: {round(best_r2, 3):.3f}")

output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best MSE', 'Value': round(best_mse,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best R2', 'Value': round(best_r2,3), 'Layers': best_params_r2_layers, 'Activation': best_params_r2_actifunc, 'Solver':best_params_r2_solver}])], ignore_index=True)
output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best Pass count', 'Value': best_pass, 'Layers': best_params_pass_layers, 'Activation': best_params_pass_actifunc, 'Solver':best_params_pass_solver}])], ignore_index=True)
output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best Passrate', 'Value': round(best_passrate,2), 'Layers': best_params_passrate_layers, 'Activation': best_params_passrate_actifunc, 'Solver':best_params_passrate_solver}])], ignore_index=True)
# output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best RMSE', 'Value': round(best_rmse,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
# output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best MAE', 'Value': round(best_mae,3), 'Layers': best_params_mse_layers, 'Activation': best_params_mse_actifunc, 'Solver':best_params_mse_solver}])], ignore_index=True)
# output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Best L2', 'Value': round(best_L2,3), 'Layers': best_params_r2_layers, 'Activation': best_params_r2_actifunc, 'Solver':best_params_r2_solver}])], ignore_index=True)
output_df = pd.concat([output_df, pd.DataFrame([{'Output': 'Total Training time', 'Value': round(total_training_time,3)}])], ignore_index=True)

# Save the output DataFrame to an Excel file
output_df.to_excel('RD_NN(II)_DIA_test2_to_Excel_output_i.xlsx', index=False)
