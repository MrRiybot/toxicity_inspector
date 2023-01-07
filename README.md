# Toxicity Inspector

## How it works

we used 2 nlp-lstm models (arabic),(english) pre-trained with datasets and pipe with lang-detection model
![model diagram](https://github.com/MrRiybot/toxicity_inspector/blob/master/model_diagram.png)

##### also you can check arabic_model.ipynb and english_mode.ipynb for more details

## Accuracy
the model give %83 accuracy on arabic
and %96 on english

## Usage
toxicity_inspector.py is the combined model
also you can use toxicity_model.ipynb for usage examples
## Installation
- First git clone the project
````
git clone https://github.com/MrRiybot/toxicity_inspector.git
````
- then import module
``````
import toxicity_inspector
``````
- then initilize class
`````
model = toxicity_inspector.Toxicity_model()
# if output is less than 0.5 then its not toxic else is toxic
model.predict("لو انت ابن راجل  انزل السعودية")
``````
