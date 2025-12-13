# ITU MLOps'25 - Project Group 101 

## Description of Task
The overall task of this project is to ensure reproducability of a model in a given GitHub repository. The task was to reconstruct and reorganize the forked repository so that it runs using a dagger pipeline and GitHub workflow. The GitHub workflow should output a model artifact named 'model' which should then be used in an inference test to ensure that the correct model was trained. The repository should also follow a Cookiecutter Data Science project structure. For reference the diagram below provides an overview of the project structure: 
![Project Architecture](./docs/project-architecture.png)


## Data and File Structure
The ML algorithm identifies users on the website that are new possible customers. It handles a classification problem where: 
- Input: The collected behavior data from users.
- Output: The classification on whether they are converted into a customer.

### Data Folder
The *data* folder, originally contains the raw data, and then stores the data in the appropriate folder according to the steps in the data pipeline.

### Pipeline.go File and itu_mlops_project_101 Folder 
The `pipeline.go` file ensures that the correct requirements are installed from the `requirements.txt` after which it runs the .py files from *itu_mlops_project_101* in the following order:
1. `data_cleaning.py`: Removes unnecessary columns and rows with missing values. 
2. `data_features.py`: Scales continuous variables, one-hot encodes categorical variables, and saves the final golden training data.    
3. `model_training.py`: Trains model using logistic regression and saves trained model as pkl file in the *models* folder.  

### test_action.yml File
The `test_action.yml` file in *.github/workflows* ensures that the dagger pipeline can run directly in GitHub, that the model artifact is created, and that the test for model inference is run.    


## How to Use
Go to 'Actions' in the GitHub repository and in 'Run Dagger pipeline' press `Run workflow`. The `model.pkl` file will be accessible after running the workflow. By clicking on the job and then 'Run Model Inference Test Action', the last line should be: ✅ Inference output matches the expected output.


## Authors
Karima Mahdaoui, kmah@itu.dk

Felicia Violeta Sørensen, feso@itu.dk

Nora Zouhtane, nozo@itu.dk

## Acknowledgement 

[Dryad README file template](https://datadryad.org/docs/README.md)


