# Archivo: q_learning_ia.py

import random

class QLearningIA:
    """Agente de Aprendizaje por Refuerzo basado en Tabla Q, que se autoconfigura a partir de GameClass."""
    
    # --- Constructor Modificado para Autoconfiguración ---
    def __init__(self, mark=1, GameClass=None):
        if GameClass is None:
            raise ValueError("QLearningIA debe inicializarse con una clase de juego (GameClass).")
            
        # 1. Autoconfiguración a partir de GameClass.CONFIG o métodos estáticos
        config = GameClass.CONFIG
        
        self.mark = mark 
        self.q_table = {} 
        
        # El board_size es solo informativo aquí; usamos el tamaño del input si está disponible, 
        # sino, preguntamos a una instancia temporal.
        self.board_size = config.get("input_size", GameClass(None, None).get_board_size())
        self.game_actions = config.get("game_actions", None) # None para TicTacToe
        
        # 2. Referencias a funciones delegadas
        # El Q-LearningIA NO necesita state_to_array_func, pero sí el validador de movimientos.
        self.get_valid_moves_func = GameClass.get_valid_moves_from_state

        # 3. Parámetros de RL
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0 
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

    def __repr__(self):
        return f"I (QLearning Mark: {self.mark})"
    
    # --- MÉTODOS DE SOPORTE GENERALES (Mapeo de acciones) ---
    
    def get_network_index(self, game_action):
        """Convierte un valor de acción del juego (ej: 1, 2, 3) a un índice (0, 1, 2)."""
        if self.game_actions is None:
            return game_action # TicTacToe: acción 0 es índice 0
        return self.game_actions.index(game_action)

    def get_game_action(self, network_index):
        """Convierte un índice (0, 1, 2) a un valor de acción del juego (ej: 1, 2, 3)."""
        if self.game_actions is None:
            return network_index # TicTacToe: índice 0 es acción 0
        return self.game_actions[network_index]
    
    # --- Método Learn (Actualización de la Tabla Q) ---
    
    def learn(self, history):
        
        for old_s, game_action, reward, new_s in history:
            
            # El estado de la tabla Q es la tupla del tablero (old_s)
            old_state_key = old_s
            
            # El estado siguiente de la tabla Q es la tupla del tablero siguiente (new_s)
            new_state_key = new_s
            
            # Convertir la acción del juego al índice de la tabla Q si es necesario
            action = self.get_network_index(game_action)
            
            # Inicializar el estado en la tabla si no existe
            if old_state_key not in self.q_table:
                self.q_table[old_state_key] = {}
            
            old_q = self.q_table[old_state_key].get(action, 0)
            
            # 1. Obtener el valor Q máximo para el nuevo estado s' (max_a' Q(s', a'))
            
            # Obtenemos las acciones válidas para el estado futuro (new_s)
            valid_actions_s_prime = self.get_valid_moves_func(new_s)
            valid_indices_s_prime = [self.get_network_index(a) for a in valid_actions_s_prime]

            if new_state_key not in self.q_table or not valid_indices_s_prime:
                max_future_q = 0
            else:
                # Si el estado existe, buscamos el valor Q máximo SÓLO entre los movimientos válidos
                q_values_s_prime = self.q_table.get(new_state_key, {})
                
                # Encontrar el máximo Q entre los índices válidos
                valid_q_values = [q_values_s_prime.get(idx, 0) for idx in valid_indices_s_prime]
                max_future_q = max(valid_q_values) if valid_q_values else 0
                
            # Fórmula de Q-Learning
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
            self.q_table[old_state_key][action] = new_q

    
    # --- Método Make Move (Toma de Decisión) ---
    
    def make_move(self, state):
        board_tuple = tuple(state["board"])
        
        # Obtener la lista de acciones válidas del juego (función delegada)
        valid_moves = self.get_valid_moves_func(board_tuple)
        
        if not valid_moves:
             return None 
             
        # Exploración (Epsilon-Greedy)
        if random.random() < self.epsilon:
            move = random.choice(valid_moves) # move es el valor de acción del juego
        
        # Explotación
        else:
            if board_tuple not in self.q_table:
                self.q_table[board_tuple] = {} 

            best_q_value = -float('inf')
            best_move = random.choice(valid_moves) # Fallback

            # Convertir las acciones válidas a índices de la tabla Q
            valid_indices = [self.get_network_index(a) for a in valid_moves]
            
            # Buscar el mejor Q entre los índices válidos
            for index in valid_indices:
                q_value = self.q_table[board_tuple].get(index, 0)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    # Almacenamos el índice de la red (que luego convertimos a acción)
                    best_index = index 
            
            # Convertir el mejor índice de la red de vuelta a la acción del juego
            # Usamos best_index para la explotación si se encontró un valor Q > -inf
            if best_q_value > -float('inf'):
                move = self.get_game_action(best_index)
            else:
                # Si todos los valores Q conocidos eran 0 o menos, usamos el movimiento de fallback
                move = best_move
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return move