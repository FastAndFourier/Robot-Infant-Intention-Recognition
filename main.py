from utilities import *
import numpy as np
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#from tqdm import tqdm
import time
import matplotlib.pyplot as plt



if __name__ == "__main__":

    # Exercice 1 et 2: Classification binaire et multi-classe sur un jeu de données

    """
        2. Ajout de l'entropie dans le vecteur de fonctionnelles

        3. Avec le paramètre voiced=True, l'analyse est portée uniquement sur les segments voisés.
           Pour voiced=False, l'analyse s'effectue sur l'ensemble des segments

           Réponse: Prendre en compte uniquement les segments non-voisés mène à un faible taux de reconnaissance
                    (en dessous du random guess). Les segments voisés sont donc plus appropriés pour la détection
                    d'intention. Notons toute fois que le calcul des fonctionnelles sur tout les segments donne
                    en moyenne de meilleurs résultats que la classification sur les segments voisés seulement.

        5. On utilise ici une SVM à marge souple et noyau gaussien. Un gridSearch ou autre techniques
           d'évaluation permettrai d'obtenir une meilleure classification

        6. Pour une classification binaire, on note que les résultats dépendent du couples de label.

            Kismet:
                ('ap','at') : accuracy = 82.98 %
                ('at','p') : accuracy = 100 %
                ('ap','p') : accuracy = 94.67 %

            BabyEars:
                ('ap','at') : accuracy = 74.48 %
                ('at','p') : accuracy = 68.91 %
                ('ap','p') : accuracy = 62.50 %

            Les performances sur BabyEars sont globalement plus faible que sur Kismet.
            On peut supposer que par méconnaissance de l'interlocuteur, les humains ont tendance à exagérer
            leurs intentions lorsqu'ils parlent à un robot.

            Les résultats sur Kismet pour les labels ap et p semblent toutefois supprenant.

    """

    # Data processing :

    # pw et pr = p

    label_kismet = ["at","p"]
    f0_kismet, en_kismet, y_kismet = load_data(label_kismet,"kismet")
    y_categorical_kismet = str2categorical(y_kismet,label_kismet)
    X_kismet = transform_functional(f0_kismet,en_kismet)

    y_kismet_random = np.random.choice(label_kismet,size=np.size(y_kismet))

    label_baby = ["at","p"]
    f0_baby, en_baby, y_baby = load_data(label_baby,"baby")
    y_categorical_baby = str2categorical(y_baby,label_baby)
    X_baby = transform_functional(f0_baby,en_baby)


    # Intention detection on Kismet
    
    print("------------------------------------------------------------")
    print("Intent detection on Kismet dataset (label considered",label_kismet,")")
    model_kismet, std_scaler_kismet = detect_intention(X_kismet,y_kismet,label_kismet)

    print("------------------------------------------------------------")
    print("Intent detection on Baby dataset (label considered",label_baby,")")
    model_baby, std_scaler_baby = detect_intention(X_baby,y_baby,label_baby)

    print("Baby dataset-trained cross corpus accuracy on kismet: {:.2f} %"
          .format(model_kismet.score(std_scaler_kismet.transform(X_baby),y_baby)*100))

    print("Kismet dataset-trained cross corpus accuracy on baby: {:.2f} %"
            .format(model_baby.score(std_scaler_baby.transform(X_kismet),y_kismet)*100))

    X_pool = np.concatenate((X_baby,X_kismet))
    y_pool = np.concatenate((y_baby,y_kismet))

    print("------------------------------------------------------------")
    print("Multi corpus intent detection (label considered",label_baby,")")
    model_pool, std_scaler_pool = detect_intention(X_pool,y_pool,label_baby)


    # Infant/Robot-directed speech classification

    X_direction = np.concatenate((X_baby,X_kismet))
    y_direction = np.concatenate((np.array(['baby']*X_baby.shape[0]),
                                  np.array(['robot']*X_kismet.shape[0])))

    

    split_idx = np.random.permutation(X_direction.shape[0])
    X_dir = X_direction[split_idx]
    y_dir = y_direction[split_idx]

    #n_training = X_dir.shape[0] - X_dir.shape[0]//10
    #print("Number of data for training:",n_training)

    print("\n---------------------------------------------------------")
    print("Baby or Robot directed speech classification")

    X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(X_dir, y_dir, test_size=0.40, random_state=42)

    scaler_dir = StandardScaler()
    X_train_dir = scaler_dir.fit_transform(X_train_dir)
    X_test_dir = scaler_dir.transform(X_test_dir)

    model_dir = SVC()
    model_dir.fit(X_train_dir,y_train_dir)
    print("Accuracy {:.2f} %".format(model_dir.score(X_test_dir,y_test_dir)*100))
    print(confusion_matrix(y_test_dir,model_dir.predict(X_test_dir)))

    
    



    