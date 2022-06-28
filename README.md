# Transformer-based Action Spotting in soccer videos
A Hierarchical Multimodal Transformer Encoder classifier is implemented in this repository to solve the task of action spotting in soccer videos.

This repository is a collection of python files structured in two folders. The first one, called one contains all the files to train the proposed model HMTAS to perform the task of action spotting. The second one, called VGGish_features, contains the files to extract audio embeddings from the soccer games.


## Model

This folder contains the following files:
 - dataset.py: file containing the required functions to generate the samples to feed the model.
 - store_data.py: file that uses dataset.py functions to generate samples and store them. 
 - model.py: file to define the implemented model (HMTAS).
 - loss.py: file that defines the loss function, KLLL with weights.
 - train.py: file that defines the functions to fit the model and do the spotting.
 - axesparraguera_main.py: main file that uses the previous ones to read data, fit the model and make predictions.
 - data_exploration.py: file to find the number of times that each action occurs in the dataset.

## VGGish_features

This folder contains the following files:
 - mel_features.py: file that defines functions to generate mel features from .wav files.
 - vggish_dataset.py: file that defines functions to generate samples to feed the VGGish model.
 - vggish_input.py: file to generate samples.
 - vggish_torch.py: file that define the VGGish model in torch.
 - vggish_params.py: file that defines the parameters of the VGGish model.
 - vggish_train.py: file that defines the functions to fit the model.
 - loss.py: file that defines the loss function.
 - vggish_post.py: file to extract the embeddings from the trained model.


Initial code inspired in [SoccerNet](https://github.com/SilvioGiancola/SoccerNetv2-DevKit) and [harritaylor](https://github.com/harritaylor/torchvggish).


## Contact  


 Mail: arturxe@gmail.com

### BibTex reference format for citation for the Code
```
@misc{HMTAS,
title={Transformer-based Action Spotting in soccer videos},
url={https://github.com/arturxe2/TFM-Artur-Xarles},
author={Artur Xarles},
  year={2022}
}
```
### BibTex reference format for citation for the report of the Master's Thesis

```
@misc{HMTAS,
title={Transformer-based Action Spotting in soccer videos},
author={Artur Xarles},
  year={2022}
}
```

