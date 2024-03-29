class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons,
            n_categories,
            epochs,
            batch_size,
            eta,
            lmbd,
            act):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.act = act
        
        self.cost_out = list()

        self.create_biases_and_weights()
        

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        if self.act == 'sigmoid':
            self.a_h = sigmoid(self.z_h)
    
            self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
            self.a_o = sigmoid(self.z_o)
        
        if self.act == 'tanh':
            self.a_h = np.tanh(self.z_h)
            
            self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
            self.a_o = np.tanh(self.z_o)
            
        #exp_term = np.exp(self.z_o)
        #print('exp_term SUM', np.sum(exp_term, keepdims=True))
        #self.probabilities = exp_term / np.sum(exp_term, keepdims=True)#, axis=1, keepdims=True)
        #print(self.probabilities)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        #a_h = sigmoid(z_h)
        
        #z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        #a_o = sigmoid(z_o)
        
        if self.act == 'sigmoid':
            a_h = sigmoid(z_h)
    
            z_o = np.matmul(a_h, self.output_weights) + self.output_bias
            a_o = sigmoid(z_o)
        
        if self.act == 'tanh':
            a_h = np.tanh(z_h)
            
            z_o = np.matmul(a_h, self.output_weights) + self.output_bias
            a_o = np.tanh(z_o)

        return a_o

    def loss_function(self):
        pass
    
    def backpropagation(self):
        #error_output = self.probabilities - self.Y_data
        error_output = (self.a_o - self.Y_data)*sigmoid_derivative(self.a_o)
        
        # ERROR M� ENDRES FOR FRANKE
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient
        
        #print('here, have updated by', self.eta * self.output_weights_gradient)

        
    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        #print(probabilities)
        #return np.argmax(probabilities, axis=1)
        #self.accuracy()
        
        probabilities[probabilities >= 0.5] = 1
        probabilities[probabilities < 0.5] = 0
        
        return probabilities
        
    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=True
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                
                self.feed_forward()
                self.backpropagation()
                
            #self.cost_function()   
        
    def printing(self):
        #print('prob', self.probabilities)
        print('O_w',  self.output_weights)
        print('O_b',  self.output_bias)
        print('H_w',  self.hidden_weights)
        print('H_B',  self.hidden_bias)
        

    # COST FUNCTION FRANKE