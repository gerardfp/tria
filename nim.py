import numpy as np
from utils import log

class Nim():
    """Implementa las reglas del juego 'Ataque del 10'. El que resta el último número gana."""

    MAX_NIM = 7
    ACTIONS = [1, 2, 3] # Movimientos permitidos: restar 1, 2, o 3

    CONFIG = {
        "input_size": 1, 
        "output_size": len(ACTIONS),
        "hidden_size": 64,      
    }

    def __init__(self, p1, p2, quiet=False):
        p1.mark = 1
        p2.mark = -1
        self.board = [self.MAX_NIM]
        self.current_player = p1
        self.other_player = p2
        self.winner = None
        self.quiet = quiet
        self.log = log.__get__(self)

    def next(self):
        while True:
            try:
                self.log(f"--- 🧸 Turno: {self.current_player}; Quedan: {self.board[0]};  Moves: {[move for move in self.ACTIONS if move <= self.board[0]]}")
                move = int(self.current_player.make_move(self.board))
                
                if 0 <= move < len(self.ACTIONS) and self.ACTIONS[move] <= self.board[0]:
                    self.process(self.ACTIONS[move]) 
                    return move
                
                self.log(f"--- ⛔️ Movimiento inválido. Debes restar {[move for move in self.ACTIONS if move <= self.board[0]]} y no exceder {self.board[0]}.")
            except ValueError:
                self.log(f"--- ⛔️ Entrada inválida. Usa {[move for move in self.ACTIONS if move <= self.board[0]]}.")
    
    def process(self, move_value):
        if move_value in self.ACTIONS and move_value <= self.board[0]:
            self.board[0] -= move_value
            
            if self.board[0] == 0:
                self.winner = self.current_player
            else:
                self.current_player, self.other_player = self.other_player, self.current_player

    def print_board(self):
        # print(f"--- 🌈 Número restante: {self.board[0]}")
        pass

    @staticmethod
    def get_valid_moves_from_state(state_board):
        return [i for i, move in enumerate(Nim.ACTIONS) if move <= state_board[0]]

    @staticmethod
    def state_to_array(board):
        """Convierte el tablero (un solo número) en un vector."""
        return np.array(board).reshape(1, -1)