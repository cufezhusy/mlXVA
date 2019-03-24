# Introduction

The repository contain the source code for the research project `Applying machin learning for troubleshooting credit exposure and XVA profiles` conducted in 2018 and 2019.

At this moment, it only contain the Convolutional Neural Network work,

# Model training

The model is trained using the GPU platform offered by `https://www.floydhub.com/`.

Training code job can be found in `\jobs`

Data used in the training are also uploaded to FloydHub, 

One can use the below command to start the model training(it took approx four hours for a standard GPU offered by FloyHub) 

```
floyd run --gpu --env tensorflow-1.11 --data cufezhusy/datasets/gengduoshuju/4:gengduoshuju 'python model_training.py'
```

Also if you have enough local computational power, can also try to run it locally.

The training results can be found in `\train_result`

# Working paper

A working paper about this is under work and will be published soon in 2019.