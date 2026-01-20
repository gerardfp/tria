import numpy as np
from utils import log
    
class TicTacToe():
    """Implementa las reglas del juego Tres-en-raya (3x3)."""
    win_positions =[
        [0, 1, 2], [3, 4, 5], [6, 7, 8], 
        [0, 3, 6], [1, 4, 7], [2, 5, 8], 
        [0, 4, 8], [2, 4, 6],           
    ]
    
    CONFIG = {
        "input_size": 9,
        "output_size": 9,
        "hidden_size": 128,
        "game_actions": None # Las acciones son los índices 0-8 (por defecto)
    }

    def __init__(self, p1, p2, quiet=False):
        p1.mark = 1
        p2.mark = -1
        self.board = [0] * 9
        self.current_player = p1
        self.other_player = p2
        self.winner = None
        self.quiet = quiet
        self.log = log.__get__(self)

    def next(self): 
        while True:
            try:
                self.log(f"\n--- Turno  {self.current_player} (Marca: {'O' if self.current_player.mark == -1 else 'X'}) ---")
                move = int(self.current_player.make_move(self.board))

                if 0 <= move < 9 and self.board[move] == 0:
                    self.process(move)
                    return move
                
                self.log("Movimiento inválido o celda ocupada. Intenta de nuevo.")
            except ValueError:
                self.log("Entrada inválida. Usa números del 1 al 9.")

    def _win(self):
        for win_pos in self.win_positions:
            marks = [self.board[p] for p in win_pos]
            if marks[0] != 0 and all(mark == marks[0] for mark in marks):
                return 1 
        
        if 0 not in self.board:
            return -1 
            
        return 0 
    
    def process(self, cmd):
        p = cmd
        if cmd is not None and 0 <= p < 9 and self.board[p] == 0:
            self.board[p] = self.current_player.mark
            win = self._win()
            
            if win == -1:
                self.winner = "draw"
            elif win != 0:
                self.winner = self.current_player
            else:
                self.current_player, self.other_player = self.other_player, self.current_player

    def print_board(self):
        self.log("-------------")
        for i in range(3):
            row = [str(cell).replace('1', 'X').replace('-1', 'O').replace('0', ' ') for cell in self.board[i*3:(i*3)+3]]
            self.log(f"| {row[0]} | {row[1]} | {row[2]} |")
            if i < 2:
                self.log("-------------")
        self.log("-------------")

    @staticmethod
    def get_valid_moves_from_state(state_board):
        """Retorna una lista de índices de movimientos válidos (0-8) a partir de un estado de tablero."""
        return [i for i in range(9) if state_board[i] == 0]

    @staticmethod
    def state_to_array(board):
        """Convierte el tablero (lista de 9) en un array de numpy (1, 9)."""
        return np.array(board).reshape(1, -1)
