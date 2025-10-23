import numpy as np
import streamlit as st

'''class MLP:
    def __init__(self,input_size=64,hidden_size=64,output_size=10,lr=0.01):
        np.random.seed(42)
        self.W1=np.random.randn(input_size,hidden_size)*0.01
        self.W2=np.random.randn(hidden_size,output_size)*0.01
        self.B1=np.zeros((1,hidden_size))
        self.B2=np.zeros((1,output_size))
        self.lr=lr
        self.actual_losses=[]
        self.epochs=3000

    def relu(self,x):
        return np.maximum(0,x)
        
    def relu_derivatives(self,x):
        return (x>0).astype(float)
    

    def softmax(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)
    
    def forward(self,X):
        
        #Forward
        self.hidden_input=np.dot(X,self.W1)+self.B1
        self.hidden_activation=self.relu(self.hidden_input)
        self.output_input=np.dot(self.hidden_activation,self.W2)+self.B2
        self.predicted_output=self.softmax(self.output_input)
        return self.predicted_output

    def backward(self,X,y,epochs=None,verbose=True):
            if epochs is None:
                epochs=self.epochs

            for epoch in range(epochs):
                output=self.forward(X)
                loss=-np.mean(np.sum(y*np.log(output+1e-9),axis=1))
                self.actual_losses.append(loss)

                #Backward
                error_output=y-output
                
                grad_W2=np.dot(self.hidden_activation.T,error_output)
                grad_B2=np.sum(error_output,axis=0,keepdims=True)
                error_hidden=np.dot(error_output,self.W2.T)*self.relu_derivatives(self.hidden_activation)
                grad_W1=np.dot(X.T,error_hidden)
                grad_B1=np.sum(error_hidden,axis=0,keepdims=True)

                self.W1-=self.lr*grad_W1
                self.W2-=self.lr*grad_W2
                self.B1-=self.lr*grad_B1
                self.B2-=self.lr*grad_B2

                loss=np.mean((y-output)**2)
                self.actual_losses.append(loss)

                if verbose and epoch % 200 == 0:
                    try:
                        st.write(f"Epochs:{epoch}\n Loss:{self.actual_losses[-1]:.4f}")
                    except:
                        print(f"Epochs:{epoch}|Loss:{self.actual_losses[-1]:.4f}")


    def predict(self,X):
        output=self.forward(X)
        return np.argmax(output,axis=1)
    
    def save(self,filename='mlp_model_weights.npz'):
        np.savez(filename,W1=self.W1,W2=self.W2,B1=self.B1,B2=self.B2)

    def load(self,filename='mlp_model_weights.npz'):
        data=np.load(filename)
        self.W1=data['W1']
        self.W2=data['W2']
        self.B1=data['B1']
        self.B2=data['B2']
        '''

'''gradient_error_output=error_output*self.sigmoid_derivatives(output)
                hidden_output=np.dot(gradient_error_output,self.W2.T)
                gradient_hidden_output=hidden_output*self.sigmoid_derivatives(self.hidden_activation)'''

import numpy as np
import streamlit as st

class MLP:
    def __init__(self, input_size=64, hidden_size=64, output_size=10, lr=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.B1 = np.zeros((1, hidden_size))
        self.B2 = np.zeros((1, output_size))
        self.lr = lr
        self.actual_losses = []
        self.accuracies = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y, epochs=3000, verbose=True):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            output = self.forward(X)

            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            self.actual_losses.append(loss)

            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            acc = np.mean(predictions == true_labels) * 100
            self.accuracies.append(acc)

            dZ2 = output - y
            dW2 = np.dot(self.A1.T, dZ2) / n_samples
            dB2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.sigmoid_derivative(self.A1)
            dW1 = np.dot(X.T, dZ1) / n_samples
            dB1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

            self.W1 -= self.lr * dW1
            self.B1 -= self.lr * dB1
            self.W2 -= self.lr * dW2
            self.B2 -= self.lr * dB2

            if verbose and epoch % 200 == 0:
                try:
                    st.write(f"Epochs:{epoch} | Loss:{loss:.4f} | Acc:{acc:.2f}%")
                except:
                    print(f"Epochs:{epoch} | Loss:{loss:.4f} | Acc:{acc:.2f}%")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save(self, filename='mlp_model_weights.npz'):
        np.savez(filename, W1=self.W1, W2=self.W2, B1=self.B1, B2=self.B2)

    def load(self, filename='mlp_model_weights.npz'):
        data = np.load(filename)
        self.W1, self.W2, self.B1, self.B2 = data['W1'], data['W2'], data['B1'], data['B2']