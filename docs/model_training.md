# Directory to store bash-scripts to execute pytorch training pipelines.

From the root directory:
1. First, you need to make bash-script executable: `chmod u+x ./bash/convnext-base-train.sh`
2. Execute: `./bash/convnext-base-train.sh`


# Training Keras models from the Terminal
- From the main directory, run: `python training_pipelines/train_keras.py <name of the model>` from the command line interface.
- You may change the validation set size, n_epochs and many more parameters from the command line. Example: `python training_pipelines/train_keras.py model_name --val_size 0.2 --epochs 10 --batch_size 32 --group_split --augment --aug_batch 32`
- Running this model will create a log file to track the progress and see previous model runs. To access the log files and check the details about the fitted model, run  `tensorboard --logdir=logs` from the command line interface.
