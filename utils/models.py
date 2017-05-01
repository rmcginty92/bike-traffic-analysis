import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM


def get_regressors(class_weights='balanced',random_state=1223):
    regressors = {
        'BayesianRidge':BayesianRidge(n_iter=500,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6),
        'OLS_L2_Ridge':Ridge(alpha=1.0),
        'Lasso':Lasso(alpha=1.0,random_state=random_state),
        'ElasticNet':ElasticNet(l1_ratio=0.15,random_state=random_state),
        'SVR':SVR(kernel='rbf',C=1.0),
        'RandomForestRegressor':RandomForestRegressor(n_estimators=50,max_depth=6),
        'ExtraTreesRegressor':ExtraTreesRegressor(n_estimators=50,max_depth=6)
    }


def lstm_base_model(input_dim,hidden_layers=None,timesteps=1,dropout_perc=0.5,**kwargs):
    output_dim = 1
    # Setting up parameters
    if hidden_layers is None:
        hidden_layers = []
    elif not isinstance(hidden_layers, (list, np.ndarray)):
        hidden_layers = [hidden_layers]
    else:
        if isinstance(hidden_layers, np.ndarray):
            layers = hidden_layers.tolist()
        else:
            layers = hidden_layers
    if any([layer_shape < 1 or isinstance(layer_shape, float) for layer_shape in
            hidden_layers]):  # layer shapes as percentages.
        hidden_layers = [int(input_dim * layer_shape) for layer_shape in hidden_layers]
    apply_dropout = dropout_perc > 0


    activation = kwargs.setdefault('activation','relu')
    init = kwargs.setdefault('init','normal')
    optimizer = kwargs.setdefault('optimizer','rmsprop')
    loss = kwargs.setdefault('loss','mean_squared_error')

    lstm_model = Sequential()
    input_shape = (timesteps, input_dim)

    for i,layer_outp in enumerate(hidden_layers):
        if i == 0:
            lstm_model.add(LSTM(layer_outp, return_sequences=True, input_shape=input_shape,
                                init=init, activation=activation))
            input_shape = (timesteps,layer_outp)
        elif i < len(hidden_layers) - 1:
            if apply_dropout and dropout_perc > 0: lstm_model.add(Dropout(dropout_perc))
            lstm_model.add(LSTM(layer_outp, return_sequences=True, input_shape=input_shape))
            input_shape = (timesteps,layer_outp)
        else: # last hidden layer
            if apply_dropout and dropout_perc > 0: lstm_model.add(Dropout(dropout_perc))
            lstm_model.add(LSTM(layer_outp, return_sequences=False, input_shape=input_shape))
            input_shape = (layer_outp,)
    lstm_model.add(Dense(output_dim))
    lstm_model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    return lstm_model


def lstm_base_model2(input_dim, hidden_layers=None, timesteps=1, dropout_perc=0.5, **kwargs):
    output_dim = 1
    # Setting up parameters
    if hidden_layers is None:
        layers = []
    elif not isinstance(hidden_layers, (list, np.ndarray)):
        layers = [hidden_layers]
    else:
        if isinstance(hidden_layers, np.ndarray):
            layers = hidden_layers.tolist()
        else:
            layers = hidden_layers
    if any([layer_shape < 1 or isinstance(layer_shape, float) for layer_shape in
            layers]):  # layer shapes as percentages.
        layers = [int(input_dim * layer_shape) for layer_shape in layers]
    layers += [output_dim]
    apply_dropout = dropout_perc > 0

    activation = kwargs.setdefault('activation', 'relu')
    init = kwargs.setdefault('init', 'normal')
    optimizer = kwargs.setdefault('optimizer', 'rmsprop')
    loss = kwargs.setdefault('loss', 'mean_squared_error')

    lstm_model = Sequential()
    input_shape = (timesteps, input_dim)

    for i, layer_outp in enumerate(layers[:-1]):
        if i == 0:
            lstm_model.add(Dense(layer_outp, input_shape=input_shape, init=init, activation=activation))
            input_shape = (timesteps, layer_outp)
        elif i < len(layers) - 1:
            if apply_dropout and dropout_perc > 0: lstm_model.add(Dropout(dropout_perc))
            lstm_model.add(LSTM(layer_outp, return_sequences=True, input_shape=input_shape))
            input_shape = (timesteps, layer_outp)
        else:  # last hidden layer
            lstm_model.add(LSTM(layer_outp, return_sequences=False, input_shape=input_shape))
            input_shape = (layer_outp,)
    lstm_model.add(Dense(output_dim))
    lstm_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return lstm_model


def lstm_base_model_old(input_dim,hidden_layers=None,timesteps=3,dropout_perc=0.5):
    output_dim = 1
    if hidden_layers is None: hidden_layers = []
    num_layers = len(hidden_layers)+2
    dropout_list = [dropout_perc for i in range(num_layers - 1)]
    if any([layer_shape < 1 or isinstance(layer_shape,float) for layer_shape in hidden_layers]): # layer shapes as percentages.
        hidden_layers = [int(input_dim*layer_shape) for layer_shape in hidden_layers]
    apply_dropout = True

    lstm_model = Sequential()
    lstm_model.add(LSTM(hidden_layers[0], return_sequences=True, input_shape=(timesteps,input_dim),init='normal', activation='relu'))
    if apply_dropout and dropout_perc > 0: lstm_model.add(Dropout(dropout_list[0]))

    for i,hidden_layer_inp in enumerate(hidden_layers[:-1]):
        lstm_model.add(LSTM(hidden_layers[i+1], return_sequences=True, input_shape=(timesteps, hidden_layer_inp)))
        if apply_dropout and dropout_perc > 0: lstm_model.add(Dropout(dropout_list[i+1]))

    lstm_model.add(LSTM(hidden_layers[-1], return_sequences=False,
                  input_shape=(timesteps, hidden_layers[-1])))

    if apply_dropout: lstm_model.add(Dropout(dropout_list[-1]))

    lstm_model.add(Dense(output_dim))

    lstm_model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])

    return lstm_model


def baseline_model(input_dim,hidden_layers=None,dropout_perc=0.5):
    output_dim = 1
    # Setting up parameters
    if hidden_layers is None:
        layers = []
    elif not isinstance(hidden_layers,(list,np.ndarray)):
        layers = [hidden_layers]
    else:
        if isinstance(hidden_layers,np.ndarray):
            layers = hidden_layers.tolist()
        else:
            layers = hidden_layers
    if any([layer_shape < 1 or isinstance(layer_shape, float) for layer_shape in layers]):  # layer shapes as percentages.
        layers = [int(input_dim * layer_shape) for layer_shape in layers]
    layers+=[output_dim]

    apply_dropout = dropout_perc > 0
    input_shape = (input_dim,)
    # create model
    model = Sequential()
    for i, layer_outp in enumerate(layers):
        if i == 0:
            model.add(Dense(layer_outp, input_shape=input_shape, init='normal', activation='relu'))
            input_shape = (layer_outp,)
            continue
        if apply_dropout and dropout_perc > 0: model.add(Dropout(dropout_perc))
        model.add(Dense(layer_outp,init='normal',input_shape=input_shape))
        input_shape = (layer_outp,)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def simple_model(input_dim,dropout_perc=0.5):
    model = Sequential()
    model.add(Dense(1, input_shape=(input_dim,), init='normal', activation='relu'))


def single_layer_model(input_dim,hidden_layer=0.5,dropout_perc=0.5,**kwargs):
    activation = kwargs.setdefault('activation','relu')
    model = Sequential()
    if isinstance(hidden_layer,float): hidden_layer = input_dim*hidden_layer
    model.add(Dense(hidden_layer, input_shape=(input_dim,), init='normal', activation=activation))
    if dropout_perc > 0: model.add(Dropout(dropout_perc))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


'''
class LSTMregressor(KerasRegressor):
    def __init__(self,input_dim=10,hidden_layers=[],timesteps=3,dropout_perc=0.5):
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.hidden_layers = self._validate_hidden_layers(hidden_layers=hidden_layers)
        self.num_layers = self._validate_num_layers()
        self.dropout_list = [dropout_perc for i in range(self.num_layers - 1)]
        super(LSTMregressor,self).__init__(build_fn=self.lstm_base_model,nb_epoch=100, batch_size=5)

    def lstm_base_model(self,input_dim,hidden_layers=[],timesteps=3,dropout_perc=0.5):
        output_dim = 1
        if any([layer_shape < 1 or isinstance(layer_shape,float) for layer_shape in hidden_layers]): # layer shapes as percentages.
            hidden_layers = [int(input_dim*layer_shape) for layer_shape in hidden_layers]
        apply_dropout = True

        lstm_model = Sequential()
        lstm_model.add(LSTM(hidden_layers[0], return_sequences=True, input_shape=(timesteps,input_dim)))
        lstm_model.add(Dense(hidden_layers[0], return_sequences=True, input_shape=(timesteps, input_dim)))
        if apply_dropout: lstm_model.add(Dropout(self.dropout_list[0]))
        for i,hidden_layer_inp in enumerate(hidden_layers[:-1]):
            next_layer_input = hidden_layers[i+1]
            lstm_model.add(LSTM(hidden_layers[i+1], return_sequences=True, input_shape=(timesteps, hidden_layer_inp)))
            if apply_dropout: lstm_model.add(Dropout(self.dropout_list[i+1]))
        lstm_model.add(LSTM(hidden_layers[-1], return_sequences=False,
                      input_shape=(timesteps, hidden_layers[-1])))
        if apply_dropout: lstm_model.add(Dropout(self.dropout_list[-1]))
        lstm_model.add(Dense(output_dim))
        lstm_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        return lstm_model

    def train_test_split(self,N,n_splits=2,sequence_rng=(5,50),test_split_ratio=0.25):
        if not isinstance(N,(int,long)):
            N = len(N)
        splits = [[] for i in range(n_splits)]
        i = 0
        while i < N:
            seq_len = np.random.randint(sequence_rng[0],sequence_rng[1],size=1)
            if n_splits > 2:
                split_list = int(np.random.rand()>test_split_ratio)
            else: split_list = int(np.random.rand()*n_splits)
            splits[split_list] = np.arange(i,i+seq_len,dtype=int)
            i+=seq_len
        return splits

    def transform_data(self,X,y):
        if len(X.shape) == 2:
            # create time sequence input
            timesteps = self.timesteps
            X = X.T
            shape = X.shape[:-1] + (X.shape[-1] - timesteps + 1, timesteps)
            strides = X.strides + (X.strides[-1],)
            Xnew = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
            X = Xnew.transpose(1,2,0)
        return X,y

    def fit(self,X,y,**fit_kwargs):
        X,y = self.transform_data(X,y)
        super(LSTMregressor,self).fit(X,y,**fit_kwargs)

    def _validate_num_layers(self):
        return len(self.hidden_layers) + 2

    def _validate_hidden_layers(self,hidden_layers):
        if isinstance(hidden_layers, (int, float, long)):
            hidden_layers = [hidden_layers]
        elif hidden_layers is None:
            hidden_layers = []
        return hidden_layers
'''

if __name__=="__main__":
    pass
