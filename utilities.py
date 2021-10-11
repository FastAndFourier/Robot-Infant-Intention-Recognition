from ast import parse
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def parse_name(dataset:list,en_path:list,path:list):

    if dataset=="kismet":
        fname = en_path.split("\\")[-1].split(".en")[0]
        parsed_label = fname[::-1][:2][::-1]
        f0_path = path + "\\" + fname + ".f0"
    elif dataset=="baby":
        fname = en_path.split("\\")[-1].split(".en")[0]
        parsed_label = fname.split(".")[-2]
        f0_path = path + "\\" + fname + ".f0"
    else:
        print("Unknown dataset")
        return None
    
    return parsed_label, f0_path

def load_data(label_list:list,dataset:list):

    if dataset == "kismet":
        path = os.getcwd() + "\\Kismet_data"
    elif dataset == "baby":
        path = os.getcwd() + "\\BabyEars_Wav"

    f0 = []
    en = []
    label = []



    for en_path in glob.glob(os.path.join(path,'*.en')):
        
        #fname = en_path.split("\\")[-1].split(".en")[0]
        #f0_path = path + "\\" + fname + ".f0"
        parsed_label, f0_path = parse_name(dataset,en_path,path)        

        if parsed_label in ["pr","pw"]:
            parsed_label = "p"

        if parsed_label not in label_list:
            continue

        label.append(parsed_label)

        data_f0 = []
        data_en = []

        with open(en_path,'r') as f:

            for x in f:
                x = x.split(" ")
                data_en.append(int(x[1]))

        with open(f0_path,'r') as f:
            
            for x in f:
                x = x.split(" ")
                data_f0.append(int(x[1]))

        f0.append(data_f0)
        en.append(data_en)
        
    f0 = np.array(f0,dtype=list)
    en = np.array(en,dtype=list)
    label = np.array(label)

    return f0, en, label

def functional(data:list):

    data = np.array(data)
    p = np.histogram(data,bins=np.unique(data))
    soft = p[0]/np.size(p)



    derivative = []
    for k in range(np.size(data)-1):
        derivative.append(abs(data[k]-data[k+1]))
    
    derivative = np.array(derivative).mean()

    return [data.mean(),data.max(),data.max()-data.min(),data.std(),np.median(data),
            np.percentile(data,25),np.percentile(data,75),(soft*np.log(soft)).sum(),derivative] 

def get_voiced_data(data_f0:list,data_en:list):
    voiced_index = np.argwhere(np.array(data_f0)!=0).T[0]
    return np.array(data_f0)[voiced_index], np.array(data_en)[voiced_index]

def transform_functional(f0:list,en:list,voiced=True):

    n_obs = np.size(f0)
    X = np.zeros((n_obs,18),dtype=float)

    for n in range(n_obs):

        if voiced:
            voiced_f0, voiced_en = get_voiced_data(f0[n],en[n])
        else:
            #print("IN")
            voiced_f0 = np.array(f0[n])
            voiced_en = np.array(en[n])

        #print(voiced_f0)
        #print(voiced_f0.mean())

        func_f0 = functional(voiced_f0)
        func_en = functional(voiced_en)

        X[n] = np.concatenate((func_f0,func_en))

    return X

def detect_intention(X:np.ndarray,y:np.ndarray,label:list):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

    label_str = categorical2label(label)
    print("Dataset balance :",end=" ")
    for l,str in zip(label_str,label):
        print("Number of",l,"= {:.2f} %".format(len(np.where(y==str)[0])/np.size(y)*100),end=" ")


    print(" ")
    std_scale = StandardScaler()
    X_train = std_scale.fit_transform(X_train)
    X_test = std_scale.transform(X_test)

    #y_train_cat = str2categorical(y_train,label)
    #y_test_cat = str2categorical(y_test,label)

    model = SVC()    
    model.fit(X_train,y_train)

    print("Accuracy SVM {:.2f} %".format(model.score(X_test,y_test)*100))
    print("Confusion matrix =\n",confusion_matrix(y_test,model.predict(X_test)))

    return model, std_scale

def categorical2label(label:np.array):

    res = np.zeros(np.size(label),dtype=list)

    for k,lb in enumerate(label):
        if lb=='ap':
            res[k] = 'Approval'
        elif lb=='at':
            res[k] = 'Attention'
        else:
            res[k] = 'Prohibition'

    return res

def str2categorical(label:np.ndarray,label_list:list):

    label_list = np.array(label_list)
    res = np.zeros((np.size(label),np.size(label_list)),dtype=int)
    
    for k in range(np.size(label)):
        res[k,:] = (label[k]==label_list)

    #print(res)
    return res
