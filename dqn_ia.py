# Archivo: dqn_ia.py

import numpy as np
import random

class SimpleNeuralNetwork:
    """Red Neuronal simple de 3 capas implementada con NumPy."""
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos (W1, b1, W2, b2)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.cache = {} 
        
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
        
    def forward(self, X):
        #print("Forward pass de la red neuronal", X)
        # Propagación hacia adelante
        Z1 = X.dot(self.W1) + self.b1 
        A1 = self.relu(Z1)            
        Z2 = A1.dot(self.W2) + self.b2 
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2}
        return Z2 # Q_values
    
    def backward(self, Q_predicted, Q_target, learning_rate):
        # Retropropagación (Descenso de Gradiente)
        X, Z1, A1, Z2 = self.cache['X'], self.cache['Z1'], self.cache['A1'], self.cache['Z2']
        m = Q_predicted.shape[0] 
        
        dLoss = Q_predicted - Q_target 

        # Capa 2 (Salida)
        dZ2 = dLoss 
        dW2 = A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Capa 1 (Oculta)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = self.relu_derivative(dA1, Z1)
        
        dW1 = X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Actualización de pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        loss = np.mean((Q_predicted - Q_target)**2)
        return loss

class DQN_IA:
    """Agente DQN generalizado que se configura a partir de la clase Game (Delegación de Responsabilidades)."""
    
    # --- Constructor Modificado ---
    def __init__(self, mark=1, GameClass=None):
          
        # 1. Autoconfiguración a partir de GameClass.CONFIG
        config = GameClass.CONFIG
        
        self.mark = mark
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.game_actions = config.get("game_actions", None) # Ej: [1, 2, 3] para Nim
        
        # 2. Referencias a funciones delegadas (el agente no conoce la lógica del juego, solo la llama)
        #self.state_to_array_func = GameClass.state_to_array 
        self.get_valid_moves_func = GameClass.get_valid_moves_from_state
        
        # 3. Inicialización de la red
        self.q_network = SimpleNeuralNetwork(
            input_size=self.input_size, 
            hidden_size=config.get("hidden_size", 128),
            output_size=self.output_size
        ) 
        
        # 4. Parámetros de RL
        self.learning_rate = 0.001 
        self.discount_factor = 0.9 
        self.epsilon = 1.0       
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

    def __repr__(self):
        return "DQN-IA"
    
    def state_to_array(self, board):
        """Codificación por defecto: lista de Python -> array de NumPy."""
        return np.array(board).reshape(1, -1)
    
    def make_move(self, state):
        board = state["board"]
        valid_moves = self.get_valid_moves_func(board)

        #print(valid_moves)
        #input("pausa")
        if not valid_moves:
             return None

        # 1. Exploración (Epsilon-Greedy)
        if random.random() < self.epsilon:
            move = random.choice(valid_moves) # move es el valor de acción del juego
            #print(f"Exploración: elijo movimiento aleatorio {move}")
        
        # 2. Explotación (Uso de Indexación Sofisticada)
        else:
            # Codificación del estado usando la función delegada
            q_values = self.q_network.forward(self.state_to_array(board))[0] 
            
            
            q_values_valid = q_values[valid_moves]
            
            relative_best_index = np.argmax(q_values_valid)
            
            best_move = valid_moves[relative_best_index]
            
            move = best_move
            #print(f"Explotación: elijo el mejor movimiento {move} con Q-valor {q_values[move]:.4f}")
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return move

    # --- Método Learn (Retropropagación) ---
    
    def learn(self, history):
        if not history:
            return
        
        # Preparar los datos usando la función de codificación específica
        old_states = np.vstack([self.state_to_array(h[0]) for h in history])
        new_states = np.vstack([self.state_to_array(h[3]) for h in history])
        
        Q_predicted_all = self.q_network.forward(old_states) 
        Q_next_all = self.q_network.forward(new_states)
        Q_target_all = Q_predicted_all.copy()
        
        for i, (old_s, action, reward, new_s) in enumerate(history):
            
            # 1. Obtener los movimientos válidos para S' (estado futuro) usando la función delegada
            valid_indices = self.get_valid_moves_func(new_s)

            #print("valid_indices:", valid_indices)

            if valid_indices:
                # Buscamos el valor Q máximo predicho por la red (Q_next_all[i]) 
                # SÓLO entre los índices que son movimientos válidos (valid_indices)
                max_future_q = Q_next_all[i, valid_indices].max()
            else:
                # Si no hay movimientos válidos (estado terminal: ganó/perdió/empató), el valor futuro es 0.
                max_future_q = 0

            # 2. Q objetivo: r + gamma * max_a'(Q(s', a'))
            target_q = reward + self.discount_factor * max_future_q
            #print("action:", action, " target_q:", target_q)
            Q_target_all[i, action] = target_q
        
        # 4. Retropropagación
        self.q_network.backward(Q_predicted_all, Q_target_all, self.learning_rate)