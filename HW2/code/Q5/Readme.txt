Information and results are present in the 	Q5report pdf

leNet.ipynb is for leNet model
ResNet.ipynb is for ResNet model

Implemented using colab with gpu enabled

Data  is accessed from '/content/gdrive/My Drive/hw2/code/' and '/content/gdrive/My Drive/hw2/Data/' folders in Mydrive after mounting

Dataloader class is used to load data with 3 objects inside: Data, target, NextLetter (used to calculate word accuracy). Removed padding from Class

for LBFGS , we have to add closure to optimizer.step().

Predicted results in predicted letter, actual letter. trainingepoc, testingepoc, wtrainingepoc, wtestingepoc are lists containing epoc related info. (letter, word accuracies.)



