Hi, we are Jiyun, Tessel, Giorgio and Alyssa, and this is the TM Group 3's github repository.
We are working on negation cue detection.

Here, you will find:
- the annotations on negation cues
- the code to perform feature extraction on our datasets
- the code to run two classifiers
- the code to perform an ablation study with those two classifiers

### CODE INSTRUCTIONS ###

Keep the original folder system to ensure correct execution of the code:
- folder: Group3_TMWork
	- subfolder: code
		- utils.py
		- preprocess.py
		- classifier_system.py
		- evaluations.py
	- subfolder: models
		- (optional) subfolders containing specific model runs

### FILES ###

Our project code consist of five files:
- preprocess.py to preprocess the corpus (SEM 2012 Shared Task Corpus on Conan Doyle stories)
- classifier_system.py: the classifier
- evaluation.py: code to compute evaluation scores
- sample_analysis.py: code to extract samples for error analysis
- utils.py: a script with helper functions

-------preprocess.py ----------
Preprocesses the file and adds the features.
Instructions: run from commandline as python preprocess.py [path]
[path] = path to the file to preprocess (train, dev or testset) on your local machine

-------classifier_system.py ----------
The classifier script. Generates model predictions on dev/test set in CONLL format files.

Instructions: run from commandline as python classifier_system.py [train_path] [dev/test_path] [output_path] [feature_ablation] [embeddings]
[train_path] path to training file on your local machine
[dev\test_path] path to development/test set on which you want to test your model
[output_path] desired path to output file (the .txt that contains predictions made by the model)
[feature_ablation] True/False if you want to do a feature ablation study or just run the model with all features
[embeddings] True/False if you also want to train the model on word embedding representations (this was made optional to save loading time)

----------evaluations.py -------------
Prints evaluation report (P+R+F1 score) and confusion matrix for files in folder. Expects a folder containing model files that can be evaluated against the same gold standard.
Instructions: run from commandline as python sample_analysis.py [path_to_gold] [path_to_folder]
[path_to_gold] = path to gold file on your local machine
[path_to_folder] = path to folder where the models that should be evaluated are situated


--------sample_analysis.py ----------
Generates a CONLL format file that can be used for error analysis. Outputs a list of all wrong classifications and aligns their features, gold label and prediction.
Instructions: run from commandline as python sample_analysis.py [path_to_gold] [path_to_predictions] [output_path]
[path_to_gold] = path to the gold file on your local machine
path_to_predictions] = path to model prediction file on your local machine
[output_path] = the desired output path where to save the sample analysis file
