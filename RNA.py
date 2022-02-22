import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1 + np.exp(-x))

dataSetColumns = [ 'Area', 'NumAmostra', 'Referencia', 'Output1', 'Output2' ] 
learnrate = 0.1
epochs = 10000
N_layers = [len(dataSetColumns) -2,4,2]

print("N_: {}".format(N_layers))
print("learnrate: ", learnrate)
print("epochs: ", epochs)

DataSet = pd.read_csv('arruela_.csv')
DataSet.drop(['Hora', 'Tamanho', 'Delta'],axis=1,inplace=True)

scaler = StandardScaler()
DataScaled = scaler.fit_transform(DataSet)
DataSetScaled = pd.DataFrame(np.array(DataScaled),columns = dataSetColumns)

X = DataSetScaled.drop(['Output1', 'Output2'],axis=1) # in
y = DataSet[['Output1','Output2']] # out

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
n_records, n_features = X_train.shape

weights_input_hidden = np.random.normal(0, scale=0.1, size=(N_layers[0], N_layers[1])) 
weights_hidden_output = np.random.normal(0, scale=0.1, size=(N_layers[1], N_layers[2]))

last_loss = None
EvolucaoError = []
IndiceError = []

for e in range(epochs):
     delta_w_i_h = np.zeros(weights_input_hidden.shape)
     delta_w_o_h = np.zeros(weights_hidden_output.shape)
     for xi, yi in zip(X_train.values, y_train.values):
          hidden_layer_input = np.dot(xi, weights_input_hidden)
          hidden_layer_output = sigmoid(hidden_layer_input)
          output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
          output = sigmoid(output_layer_in)

          output_error_term = (yi - output) * output * (1 - output)
          hidden_error = np.dot(weights_hidden_output,output_error_term)
          hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
          delta_w_o_h += output_error_term*hidden_layer_output[:, None]
          delta_w_i_h += hidden_error_term * xi[:, None]
          
     weights_input_hidden += learnrate * delta_w_i_h / n_records
     weights_hidden_output += learnrate * delta_w_o_h / n_records

     if  e % (epochs / 1000) == 0:
          hidden_output = sigmoid(np.dot(xi, weights_input_hidden))
          out = sigmoid(np.dot(hidden_output, weights_hidden_output))
          loss = np.mean((out - yi) ** 2)

          print(e, "Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando" if last_loss and last_loss < loss else "")
          last_loss = loss

          EvolucaoError.append(loss)
          IndiceError.append(e)

n_records, n_features = X_test.shape
predictions = 0

for xi, yi in zip(X_test.values, y_test.values):
     hidden_layer_input = np.dot(xi, weights_input_hidden)
     hidden_layer_output = sigmoid(hidden_layer_input)

     output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
     output = sigmoid(output_layer_in)
    
     if ((output[0]>output[1] and yi[0]>yi[1]) or (output[1]>=output[0] and yi[1]>yi[0])):
          predictions+=1
                
print("A Acurácia da Predição é de: {:.3f}".format(predictions/n_records))
 
f = open("results.txt", "a")
f.write("N_: {}\n".format(N_layers))
f.write("learnrate: {}\n".format(learnrate))
f.write("epochs: {}\n".format(epochs))
f.write("Colunas: {}\n".format(dataSetColumns))
f.write("A Acuracia da Predicao e de: {:.5f}\n".format(predictions/n_records))
f.write("\n\n")
f.close()
