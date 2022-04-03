# Classification d'images de déchets 

**Cadre du projet :** Cours de Deep Learning, mention Intelligence Artificielle, CentraleSupélec 3ème année

**Deadline :** 10 avril 2022

## Présentation du sujet

Le but du projet est de développer un modèle d’apprentissage profond permettant de classifier des images de déchets.
L’ensemble de données utilisé provient du dataset Garbage accessible sur Kaggle : https://www.kaggle.com/asdasdasasdas/garbage-classification.

Celui-ci comporte un total de 2527 images de 6 classes différentes : papier, métal, carton, plastique, verre et autres déchets.

La stratégie du Transfer Learning a été utilisée afin d’éviter le problème de surapprentissage étant donné la petite taille du dataset. 


## Structure du projet

- `split_dataset.py` : Ce fichier est à lancer en premier. Il permet de générer un dossier Garbage_train_test_val ayant cette structure :
```
|_ Garbage_train_test_val/
   |_ train/
      |_ paper/
      |_ metal/
      |_ .../
   |_ test/
      |_ paper/
      |_ metal/
      |_ .../
   |_ validation/
      |_ paper/
      |_ metal/
      |_ .../
```


- `transfer_learning.py` : Ce fichier permet d'adapter et tester les 11 modèles de classification d'images pré-entraînés VGG16, VGG19, ResNet50, Xception, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, NASNetLarge et NASNetMobile.

- `model_comparison.py` : Afin de comparer les performances des modèles ci-dessus en termes d'accuracy, rappel, précision, score-f1, nombre de paramètres et durée d'entraînement. Les résultats sont présentés dans le fichier `./Results/transfer_learning_compare_models.csv`


Par ailleurs, les différents notebooks placés dans le dossier du même nom retracent les différentes étapes du projet de façon chronologique et présentent différentes visualisations : les images après augmentation, les évolutions de l'accuracy et de la loss des modèles entraînés au fil des epochs, ...
