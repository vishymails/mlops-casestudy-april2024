
Create Environment 

```
conda create -n wineq python=3.8 -y
conda activate wineq
```

Create requirements.txt


```
dvc
dvc[gdrive]
scikit-learn

```

```
pip install -r requirements.txt

```


Create template.py 

```
import os

dirs = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src",
    "data_given"
]

for dir_ in dirs :
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f :
        pass

files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("src", "__init__.py")
]


for file_ in files :
    with open(file_, "w") as f :
        pass

```

```
python template.py

Observe project directory structure has been created 


```

```
Download data file (winequality.csv) from 

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5

copy winequality.csv to data_given folder 

```

```
dvc init
dvc add data_given/winequality.csv
git add .
git commit -m "csv commit"

git remote add origin https://github.com/vishymails/mlops-casestudy-april2024.git

git push -u origin main

```


Update params.yaml file 

```
base :
  project : winequality-project
  random_state : 42
  target_col : TARGET

data_source : 
  s3_source : data_given/winequality.csv

load_data :
  raw_dataset_csv : data/raw/winequality.csv

split_data :
  train_path : data/processed/train_winequality.csv
  test_path : data/processed/test_winequality.csv
  test_size : 0.2

estimators :
  ElasticNet :
    params :
      # alpha : 0.88
      # l1_ratio : 0.11
      # alpha : 0.9
      # l1_ratio : 0.4
      alpha : 0.2
      l1_ratio : 0.2

model_dir : saved_models

reports :
  params : report/params.json
  scores : report/scores.json

webapp_model_dir : prediction_service/model/model.joblib

mlflow_config :
  artifacts_dir : artifacts
  experiment_name : ElasticNet regression
  run_name : mlops
  registered_model_name : ElasticNetWineModel
  remote_server_uri : http://0.0.0.0:1234

  ```

  ```
  git add .
git commit -m "commited1"
git push -u origin main
```



Update requirements.txt

```
dvc
dvc[gdrive]
scikit-learn
pandas
pytest
tox
flake8
flask
gunicorn
mlflow



```

```

pip install -r requirements.txt

```

create src/get_data.py

```
import os
import yaml
import pandas as pd
import argparse

def read_params(config_path) :
    with open(config_path) as yaml_file :
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path) :
    config = read_params(config_path)

    print(config)

    data_path = config["data_source"]["s3_source"]

    df = pd.read_csv(data_path, sep=",", encoding="utf-8")

    print(df)

    return df


if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
```

Create load_data.py 

```

import os
from get_data import read_params, get_data
import argparse

def load_and_save(config_path) :
    config = read_params(config_path)
    df = get_data(config_path)

    new_cols = [col.replace(" ", "_") for col in df.columns]
    print(new_cols)

    raw_data_path = config["load_data"]["raw_dataset_csv"]

    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)


if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)



```


Add stage to dvc.yaml file 

```
stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

```

```
dvc repro
```


Create split_data.py

```
import os
import argparse 
import pandas as pd 
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_and_saved_data(config_path) :
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path, sep=",")

    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)

    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)


```


Update dvc.yaml 

```

stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

  split_data :
    cmd : python src/split_data.py --config=params.yaml
    deps :
      - src/split_data.py
      - data/raw/winequality.csv
    outs :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv

      
```

```
dvc repro
```


Create report directory and its files : params.json, scores.json

```
mkdir report
touch report/params.json
touch report/scores.json
```

Write train_and_evaluate.py

```

import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
from urllib.parse import urlparse

import argparse 
import joblib
import json


def eval_metrics(actual, pred) :
    rmse = np.sqrt(mean_absolute_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path) :
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]  

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]


    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)

    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("ElasticNet model (alpha=%f, l1_ratio=%f): " % (alpha, l1_ratio))

    print("RMSE : %s" % rmse)
    print("MAE : %s" % mae)
    print("R2 : %s" % r2)

    # Store above generated data for reporting and metrics calculation 

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f :
        scores = {
            "rmse" : rmse,
            "mae": mae,
            "r2": r2
        }

        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f :
        params = {
            "alpha" : alpha,
            "l1_ratio": l1_ratio
        }

        json.dump(params, f, indent=4)


    # dump model as joblib for further usage 

    os.makedirs(model_dir, exist_ok = True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)
    
    




if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)




```


Update dvc.yaml

```
stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

  split_data :
    cmd : python src/split_data.py --config=params.yaml
    deps :
      - src/split_data.py
      - data/raw/winequality.csv
    outs :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv

  train_and_evaluate :
    cmd : python src/train_and_evaluate.py --config=params.yaml
    deps :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv
      - src/train_and_evaluate.py
    params :
      - estimators.ElasticNet.params.alpha
      - estimators.ElasticNet.params.l1_ratio
    metrics :
      - report/scores.json :
          cache : false
      - report/params.json :
          cache: false
    outs :
      - saved_models/model.joblib

      

```

```
dvc repro

```
```
git add .
git commit -m "param changes "
git push -u origin main
```

View Current metrics 

```
dvc repro
dvc params diff
dvc metrics show
dvc metrics diff

You may get blank values in some commands because there are no changes in parameters 
```

Update params.yaml file 

```
estimators :
  ElasticNet :
    params :
      alpha : 0.88
      l1_ratio : 0.11
      # alpha : 0.9
      # l1_ratio : 0.4
      #alpha : 0.2
      #l1_ratio : 0.2
```
```
git add .
git commit -m "param changes "
git push -u origin main
```

```
dvc repro
dvc params diff
dvc metrics show
dvc metrics diff

Observe the changes 
```

```
git add .
git commit -m "param changes "
git push -u origin main
```


# Now Pipeline tasks are done Lets test using pytest and TOX frameworks

```
verify pytest and tox are declared in requirements.txt
```


Create tox.ini

```
[tox]
envlist = py38
skipdist = True

[testenv]
deps = -rrequirements.txt
command = 
    pytest -v

```

```
pytest -v 
```


Create tests folder with test cases

```
mkdir tests
touch tests/conftest.py
touch tests/test_config.py
touch tests/__init__.py
```

Update tests/test_config.py

```
def test_generic() :
    a = 2
    b = 2
    assert a == b

```

```
pytest -v 
```

# Create branding using setup.py and get wheel file 

setup.py

```
from setuptools import setup, find_packages

setup(
    name = "src",
    version="0.0.1",
    description="Case study project for Oracle India",
    author="BVR",
    packages=find_packages(),
    license="MIT"
)
```


```
pip install -e .
pip freeze
```

Build your own package commands 

```
python setup.py sdist bdist_wheel
```


# Create schema using jupyter lab

```
pip install jupyterlab

jupyter-lab notebooks/

```

```
EXECUTE ALL CONTENTS PRESENT IN THE NOTEBOOK1.IPYNB 
```


# Test case examples


Update test_config.py

```
def test_generic() :
    a = 2
    b = 2
    assert a == b


def test_generic1() :
    a = 2
    b = 2
    assert a != b


class NotInRange(Exception) :
    def __init__(self, message="value not in given range- by BVR") :
        self.message = message
        super().__init__(self.message)


def test_generic2() :
    a = 15
    if a not in range(10, 20) :
        raise NotInRange
    

def test_generic3() :
    a = 5
    if a not in range(10, 20) :
        raise NotInRange
    
```

```
pytest -v
tox
```


# Create UI for created model application 

Create necessary files and folders 

```
mkdir prediction_service
  mkdir -p prediction_service/model
  mkdir webapp
  touch app.py
  touch prediction_service/__init__.py
  touch prediction_service/prediction.py
  mkdir -p webapp/static/css
  mkdir -p webapp/static/script
  touch webapp/static/css/main.css
  touch webapp/static/script/index.js
  mkdir  webapp/templates
  touch webapp/templates/index.html
  touch webapp/templates/404.html
  touch webapp/templates/base.html

```

# Copy schema_in.json file from notebooks to prediction_service folder 

# Create necessary files 

```
fill all files under webapp with saved contents
```

Create app.py (Part 1) - further file will get updated 

```
from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np 


params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods = ["GET", "POST"])
def index() :
    if request.method == "POST" :
        pass
    else :
        return render_template("index.html")
    


if __name__ == "__main__" :
    app.run(host="0.0.0.0", port=5000, debug=True)
    

```

Run Flask Development server 
```
python app.py 

```

Run on Browser 

```
http://localhost:5000
```

# Alter to serve model request

```
Copy joblib file to prediction_service/model folder 

copy schema_in.json to prediction_service folder 
```




Update app.py 

```
from flask import Flask, render_template, request, jsonify
import os
import joblib
import yaml
import numpy as np 


params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")


def read_params(config_path) :
    with open(config_path) as yaml_file :
        config = yaml.safe_load(yaml_file)
    return config


def predict(data) :
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction 


def api_response(request) :
    pass


app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods = ["GET", "POST"])
def index() :
    if request.method == "POST" :
        try :
            if request.form :
                data = dict(request.form).values()
                data = [list(map(float, data))]
                response = predict(data)
                return render_template("index.html", response=response)
            elif request.json :
                response = api_response(request)
                return jsonify(response)
        except Exception as e :
            print(e)
            error = {"error" : "Something went wrong !! Try again "}
            return render_template("404.html", error = error)
    else :
        return render_template("index.html")
    


if __name__ == "__main__" :
    app.run(host="0.0.0.0", port=5000, debug=True)
    
```

```
python app.py 

In browser http://localhost:5000

FILL ALL FIELDS WITH 12 (ANY NUMBER) AS VALUES AND SUBMIT TO SEE THE PREDICTED RESULT 

Note : validation is mandatory so we will use component architecture and validation in next steps 
```

# Perform validation of fields 

Update prediction.py 

```
import joblib
import yaml
import os
import json
import numpy as np




params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")


class NotInRange(Exception) :
    def __init__(self, message="value not in given range- by BVR") :
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception) :
    def __init__(self, message="Not in Columns") :
        self.message = message
        super().__init__(self.message)



def read_params(config_path = params_path) :
    with open(config_path) as yaml_file :
        config = yaml.safe_load(yaml_file)
    return config


def predict(data) :
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    try :
        if 3 <= prediction <=8 :
            return prediction
        else :
            raise NotInRange
    except NotInRange :
        return "UnExpected Result"


def get_schema(schema_path = schema_path) :
    with open(schema_path) as json_file :
        schema = json.load(json_file)
    return schema

def validate_input(dict_request) :
    def _validate_cols(col) :
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols :
            raise NotInCols
        
    def _validate_values(col, val) :
        schema = get_schema()
        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange
        
    for col, val in dict_request.items() :
        _validate_cols(col)
        _validate_values(col, val)

    return True 


def form_response(dict_request) :
    if validate_input(dict_request) :
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response


def api_response(request) :

    pass


```


Update app.py 

```
from flask import Flask, render_template, request, jsonify
import os
from prediction_service import prediction 
import numpy as np 


webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")



app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods = ["GET", "POST"])
def index() :
    if request.method == "POST" :
        try :
            if request.form :
                dict_req = dict(request.form)
                response = prediction.form_response(dict_req)
                return render_template("index.html", response=response)
            elif request.json :
                response = prediction.api_response(request)
                return jsonify(response)
        except Exception as e :
            print(e)
            error = {"error" : "Something went wrong !! Try again "}
            error = {"error" : e}
            return render_template("404.html", error = error)
    else :
        return render_template("index.html")
    


if __name__ == "__main__" :
    app.run(host="0.0.0.0", port=5000, debug=True)
    

```

```
python app.py 

http://localhost:5000
```



# API based microservice creation and sending request using postman


Update prediction.py 

```


def api_response(dict_request) :
    try :
        if validate_input(dict_request) :
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response" : response}
            return response 
    except NotInRange as e :
        response = {"the_expected_range" : get_schema(), "response" : str(e)}
        print(response)

    except NotInCols as e :
        response = {"the_expected_cols" : get_schema().keys(), "response" : str(e)}
        print(response)

    except Exception as e :
        print(e)
        error = {"error" : "Something went wrong !! Try Again"}
        return error

```


Update app.py 

```

@app.route("/", methods = ["GET", "POST"])
def index() :
    if request.method == "POST" :
        try :
            if request.form :
                dict_req = dict(request.form)
                response = prediction.form_response(dict_req)
                return render_template("index.html", response=response)
            elif request.json :
                response = prediction.api_response(request.json)
                return jsonify(response)
        except Exception as e :
            print(e)
            error = {"error" : "Something went wrong !! Try again "}
            error = {"error" : e}
            return render_template("404.html", error = error)
    else :
        return render_template("index.html")
    
```

```
python app.py 
```

Run post man 

```
new Request : http://localhost:5000

METHOD : POST 

TYPE : RAW / JSON 

BODY AS DEFINED 

```

Body 

```
{
    "fixed_acidity" : 4.6, 
    "volatile_acidity" : 0.12,
    "citric_acid" : 0.1,
    "residual_sugar" : 0.9,
    "chlorides" : 0.012,
    "free_sulfur_dioxide" : 1.0,
    "total_sulfur_dioxide" : 6.0,
    "density" : 0.99007,
    "pH" : 2.74,
    "sulphates" : 0.33,
    "alcohol" : 8.4
}

```


Add MLFlow Service to existing application 

Update train_and_evaluate.py 

```

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow


def eval_metrics(actual, pred) :
    rmse = np.sqrt(mean_absolute_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



def train_and_evaluate(config_path) :
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path,sep=",")
    test = pd.read_csv(test_data_path,sep=",")

    train_y = train[target] 
    test_y = test[target]


    train_x = train.drop(target, axis=1) 
    test_x = test.drop(target, axis=1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run :


        lr = ElasticNet(alpha=alpha, 
            l1_ratio=l1_ratio,
            random_state=random_state)
        lr.fit(train_x, train_y)


        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store !="file" :
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name=mlflow_config["registered_model_name"])
        else :
            mlflow.sklearn.load_model(lr, "model")        

    


if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)


```


Update dvc.yaml file 

```

stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

  split_data :
    cmd : python src/split_data.py --config=params.yaml
    deps :
      - src/split_data.py
      - data/raw/winequality.csv
    outs :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv

  train_and_evaluate :
    cmd : python src/train_and_evaluate.py --config=params.yaml
    deps :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv
      - src/train_and_evaluate.py
    params :
      - estimators.ElasticNet.params.alpha
      - estimators.ElasticNet.params.l1_ratio
    # metrics :
    #   - report/scores.json :
    #       cache : false
    #   - report/params.json :
    #       cache: false
    # outs :
    #   - saved_models/model.joblib

  # log_production_model :
  #   cmd : python src/log_production_model.py --config=params.yaml
  #   deps : 
  #     - src/log_production_model.py


      


```


Update params.yaml 

```

base :
  project : winequality-project
  random_state : 42
  target_col : TARGET

data_source : 
  s3_source : data_given/winequality.csv

load_data :
  raw_dataset_csv : data/raw/winequality.csv

split_data :
  train_path : data/processed/train_winequality.csv
  test_path : data/processed/test_winequality.csv
  test_size : 0.2

estimators :
  ElasticNet :
    params :
      alpha : 0.88
      l1_ratio : 0.11
      # alpha : 0.9
      # l1_ratio : 0.4
      #alpha : 0.2
      #l1_ratio : 0.2

model_dir : saved_models

reports :
  params : report/params.json
  scores : report/scores.json

webapp_model_dir : prediction_service/model/model.joblib

mlflow_config :
  artifacts_dir : artifacts
  experiment_name : ElasticNetRegression
  run_name : mlops
  registered_model_name : ElasticNetWineModel
  remote_server_uri : http://localhost:1234

```

Open fresh terminal with wineq as env 

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost -p 1234

```


Run

```
dvc repro 
```

Open 

```
http://localhost:1234

```