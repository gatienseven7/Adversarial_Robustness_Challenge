# Adversarial_Robustness_Challenge

Voici un exemple de fichier **Markdown** que tu pourrais ajouter dans le **README** de ton repo GitHub, avec les informations que tu m'as fournies. J'ai ajouté des sections pour expliquer les différentes étapes du projet et les résultats obtenus, ainsi que des conclusions.

```markdown
# Adversarial Robustness Challenge - MNIST

## Objectif
Ce projet vise à entraîner un modèle de classification d'images robuste aux attaques adversariales. Nous avons utilisé un réseau de neurones convolutifs (CNN) et TensorFlow pour l'entraînement et la génération d'exemples adversariaux, puis nous avons réentraîné le modèle avec des données adversariales pour améliorer sa robustesse.

## Données
Le jeu de données utilisé est le **Kaggle - Digit Recognizer (MNIST)**, qui contient des images de chiffres manuscrits. Voici un aperçu des dimensions des données utilisées dans ce projet :

- **Shape of df**: (32696, 785)  
  Le jeu de données original contient 32 696 exemples, avec 784 caractéristiques pour chaque image (28x28 pixels) et une colonne `label` pour l'étiquette (chiffre de 0 à 9).
  
- **x_train shape**: (26156, 28, 28, 1)  
  Le jeu d'entraînement contient 26 156 images, chacune de taille 28x28 pixels et avec une dimension supplémentaire pour les canaux (1 canal pour les images en niveaux de gris).

- **x_test shape**: (6540, 28, 28, 1)  
  Le jeu de test contient 6 540 images avec la même dimension que celles de l'entraînement.

- **x_test_adv shape**: (10, 28, 28, 1)  
  Un échantillon de 10 images adversariales générées à l'aide de l'attaque **FGSM**.

## Modèle
Nous avons construit un réseau de neurones convolutif simple avec **TensorFlow** pour la classification des chiffres :

```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Optimisation et Entraînement
Le modèle a été compilé avec l'optimiseur **Adam** et la fonction de perte **categorical_crossentropy**, puis entraîné sur 5 époques.

## Attaque Adversariale (FGSM)
Nous avons généré des exemples adversariaux à l'aide de l'attaque **FGSM (Fast Gradient Sign Method)**. Un petit epsilon de 0.01 a été utilisé pour créer les perturbations sur les images de test. Les perturbations sont calculées à partir du gradient de la perte par rapport aux entrées.

## Résultats
### Entraînement du modèle sur les données normales

Les résultats de l'entraînement sur les données normales sont les suivants :

- **Précision sur les données normales (clean accuracy)** : 97.75%
- **Précision sur les données adversariales (adversarial accuracy)** : 1.07%

Voici les courbes d'entraînement et de validation pendant les 5 époques :

```
Epoch 1/5
Accuracy: 81.02%, Loss: 0.6655
Validation Accuracy: 94.65%, Validation Loss: 0.1848

Epoch 2/5
Accuracy: 96.46%, Loss: 0.1264
Validation Accuracy: 96.37%, Validation Loss: 0.1208

Epoch 3/5
Accuracy: 97.91%, Loss: 0.0730
Validation Accuracy: 97.67%, Validation Loss: 0.0915

Epoch 4/5
Accuracy: 98.22%, Loss: 0.0551
Validation Accuracy: 97.33%, Validation Loss: 0.0886

Epoch 5/5
Accuracy: 98.96%, Loss: 0.0377
Validation Accuracy: 97.59%, Validation Loss: 0.0865
```

### Précision après réentraînement avec des exemples adversariaux
Le modèle a ensuite été réentraîné en incluant des exemples adversariaux dans l'ensemble d'entraînement. Voici les résultats après l'entraînement :

- **Précision sur les données normales (clean accuracy)** : 97.75%
- **Précision sur les données adversariales (adversarial accuracy)** : 100.00%

### Conclusion sur la robustesse adversariale
Après réentraîner le modèle avec des exemples adversariaux, nous avons observé une **amélioration significative** de la précision sur les images adversariales, atteignant 100%. Cependant, la précision sur les données normales n'a pas été impactée, ce qui indique que l'entraînement adversarial a renforcé la robustesse du modèle tout en maintenant sa performance sur les données non perturbées.

## Techniques de Défense
### 1. **Entraînement Adversarial**
Le principal mécanisme de défense utilisé ici est l'**entraînement adversarial**, qui consiste à réentraîner le modèle en utilisant des exemples adversariaux. Cette technique permet d'améliorer la capacité du modèle à détecter et à traiter les perturbations adversariales. Elle a été appliquée en ajoutant les exemples adversariaux à l'ensemble d'entraînement et en réévaluant les performances sur les deux types d'exemples (normaux et adversariaux).

### 2. **Limitation des Perturbations**
Lors de la génération des exemples adversariaux, les perturbations ont été limitées à une certaine amplitude en utilisant `tf.clip_by_value(perturbations, -0.3, 0.3)` pour éviter que les images perturbées ne deviennent trop irréalistes ou inutilisables.

## Conclusion
Ce projet démontre l'importance de **prendre en compte la sécurité adversariale** dans les systèmes de machine learning, notamment dans les applications de reconnaissance d'images. L'entraînement adversarial est une technique efficace pour améliorer la robustesse d'un modèle face aux attaques adversariales, tout en maintenant des performances élevées sur des données normales.

## Liens Utiles
- [MNIST Dataset on Kaggle](https://www.kaggle.com/c/digit-recognizer)
- [TensorFlow Documentation](https://www.tensorflow.org/)
  
