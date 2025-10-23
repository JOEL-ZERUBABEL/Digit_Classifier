import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from mlp_model import MLP
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

class Train:
    def __init__(self):
        digits=load_digits()
        self.X=digits.data/16.0
        self.y=digits.target.reshape(-1,1)
        self.onehotencoder=OneHotEncoder(sparse_output=False)
        self.y_encoded=self.onehotencoder.fit_transform(self.y)
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y_encoded,test_size=0.2,random_state=42)
        
    def models(self):
        #MLPDigitClassifier input_size=64,hidden_size=32,output_size=10,lr=0.1
        mlpdigitclassifier=MLP(input_size=64,hidden_size=64,output_size=10,lr=0.05)
        mlpdigitclassifier.backward(self.X_train,self.y_train,epochs=3000,verbose=True)
        mlpdigitclassifier.save('mlp_model_weights.npz')
        print('Model trained succesfully')
        preds=mlpdigitclassifier.predict(self.X_test)
        y_test_labels=np.argmax(self.y_test,axis=1)
        acc=accuracy_score(y_test_labels,preds)
        return mlpdigitclassifier

    def plot(self,mlpdigitclassifier):
        plt.figure(figsize=(8,4))
        plt.plot(mlpdigitclassifier.actual_losses,color='blue')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()  

if __name__=='__main__':
    t=Train()
    models=t.models()
    t.plot(models)