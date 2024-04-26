from get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os



def log_production_model(config_path):
    config = read_params(config_path)
    
    
    mlflow_config = config["mlflow_config"] 
    

    model_name = mlflow_config["registered_model_name"]


    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    
    #get all experiments done 
    #we did not crete any experiment name by default it takes 1 
    runs = mlflow.search_runs(experiment_ids=1) # SEE WE NAMED THE EXPERIMENT SO DEFAULT 0 IS NOT CREATED AND TAKES 1 REFER 'artifacts' FOLDER

    lowest = runs["metrics.mae"].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"][0]
    

    client = MlflowClient() #mv - model version
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4) # prints in key value pair 
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )        


    loaded_model = mlflow.pyfunc.load_model(logged_model) #altered
    
    model_path = config["webapp_model_dir"] #"prediction_service/model"

    joblib.dump(loaded_model, model_path) # this will dump the model to model path 
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)