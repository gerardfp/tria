# Archivo: dqn_ia.py

import numpy as np
import random
from copy import deepcopy

class SimpleNeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, debug=False):
        # Inicialización de pesos y sesgos (W1, b1, W2, b2)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        self.debug = debug
        
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
        
    def forward(self, X):
        if self.debug:
            print("⏩ Forward pass de la red neuronal")
            print("Input X:")
            print(X)
        
        self.X = X
        self.Z1 = X.dot(self.W1) + self.b1 
        self.A1 = self.relu(self.Z1) 
        self.Z2 = self.A1.dot(self.W2) + self.b2

        return self.Z2 # Q_values
    
    def backward(self, Q_predicted, Q_target, learning_rate):
        m = Q_predicted.shape[0] 
        
        dLoss = Q_predicted - Q_target 

        # Capa 2 (Salida)
        dZ2 = dLoss 
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Capa 1 (Oculta)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = self.relu_derivative(dA1, self.Z1)

        dW1 = self.X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Actualización de pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


class DQN_IA:
    """Agente DQN generalizado que se configura a partir de la clase Game (Delegación de Responsabilidades)."""
    
    def __init__(self, GameClass=None, debug=False):
          
        self.debug = debug
        self.mark = None
        
        self.get_valid_moves_func = GameClass.get_valid_moves_from_state
        self.game_state_to_array_func = GameClass.state_to_array
        
        self.q_network = SimpleNeuralNetwork(
            input_size= GameClass.CONFIG["input_size"], 
            hidden_size=GameClass.CONFIG.get("hidden_size", 128),
            output_size=GameClass.CONFIG["output_size"],
            debug=debug,
        ) 
        self.target_network = deepcopy(self.q_network)
        self.target_update_freq = 10  # episodios
        self.learn_step = 0

        self.learning_rate = 0.001 
        self.discount_factor = 0.9 
        self.epsilon = 1.0       
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.01

    def __repr__(self):
        return "DQN-IA"
    
    def state_to_array(self, board):
        return self.game_state_to_array_func(board)
    
    def make_move(self, state):
        board = state["board"]
        valid_moves = self.get_valid_moves_func(board)

        if not valid_moves:
             return None
        
        # 1. Exploración (Epsilon-Greedy)
        if random.random() < self.epsilon:
            if self.debug:
                print("🎲 Exploración: eligiendo movimiento aleatorio...")

            best_move = random.choice(valid_moves) # move es el valor de acción del juego

            if self.debug:
                print(f"Move aleatorio: {best_move} e:{self.epsilon:.4f}")
            
        # 2. Explotación
        else:
            if self.debug:
                print("💥 Explotación: calculando Q values para el estado actual...")

            q_values = self.q_network.forward(self.state_to_array(board))[0] 
            
            if self.debug:
                print("valid_moves:")
                print(valid_moves)
                print("Q values:")
                print(q_values)
            
            q_values_valid = q_values[valid_moves]

            relative_best_index = np.argmax(q_values_valid)
            
            best_move = valid_moves[relative_best_index]

            if self.debug:
                print("Q values válidos:")
                print(q_values_valid)
                print(f"Best move index (relativo): {relative_best_index}")
                print(f"Best move (absoluto): {best_move}")
                print(f"Best move Q-value: {q_values[best_move]}")
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return best_move

    
    def learn(self, history):
        """ history: [old_board, move, reward, new_board] """
        if not history:
            return
        if (self.debug):
            print("🤖 DQN_IA aprendiendo de la historia de la partida...")
            print("Historia:", history)

        old_states = np.vstack([self.state_to_array(h[0]) for h in history])
        new_states = np.vstack([self.state_to_array(h[3]) for h in history])
        
        Q_predicted_all = self.q_network.forward(old_states)
        Q_next_all = self.target_network.forward(new_states)
        Q_target_all = Q_predicted_all.copy()

        if self.debug:
            print("Q_predicted_all:")
            print(Q_predicted_all)
            print("Q_target_all:")
            print(Q_target_all)
        
        for i, (old_s, action, reward, new_s) in enumerate(history):
            
            valid_indices = self.get_valid_moves_func(new_s)

            if valid_indices:
                max_future_q = Q_next_all[i, valid_indices].max()
            else:
                max_future_q = 0

            # 2. Q objetivo: r + gamma * max_a'(Q(s', a'))
            target_q = reward + self.discount_factor * max_future_q
            Q_target_all[i, action] = target_q
        
        if self.debug:
            print("Después de actualizar los Q_target_all:")
            print("Q_predicted_all:")
            print(Q_predicted_all)
            print("Q_target_all:")
            print(Q_target_all)
            input("pausa end_learn")
        self.q_network.backward(Q_predicted_all, Q_target_all, self.learning_rate)

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network = deepcopy(self.q_network)  