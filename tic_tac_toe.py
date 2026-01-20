import numpy as np
    
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
        self.board_size = 9
        p1.mark = 1
        p2.mark = -1
        self.state = {
            "board": [0] * self.board_size,
            "current_player": p1,
            "other_player": p2,
            "winner": None,
        }
        self.quiet = quiet

    def next(self): 
        while True:
            try:
                if not self.quiet:
                    print(f"\n--- Turno  {self.state["current_player"]} (Marca: {'O' if self.state["current_player"].mark == -1 else 'X'}) ---")
                move = int(self.state["current_player"].make_move(self.state))

                if 0 <= move < 9 and self.state["board"][move] == 0:
                    self.process(move)
                    return move
                elif not self.quiet:
                    print("Movimiento inválido o celda ocupada. Intenta de nuevo.")
            except ValueError:
                if not self.quiet:
                    print("Entrada inválida. Usa números del 1 al 9.")


    def _win(self):
        for win_pos in self.win_positions:
            marks = [self.state["board"][p] for p in win_pos]
            if marks[0] != 0 and all(mark == marks[0] for mark in marks):
                return 1 
        
        if 0 not in self.state["board"]:
            return -1 
            
        return 0 
    
    def process(self, cmd):
        p = cmd
        if cmd is not None and 0 <= p < self.board_size and self.state["board"][p] == 0:
            self.state["board"][p] = self.state["current_player"].mark
            win = self._win()
            
            if win == -1:
                self.state["winner"] = "draw"
            elif win != 0:
                self.state["winner"] = self.state["current_player"]
            else:
                self.state["current_player"], self.state["other_player"] = self.state["other_player"], self.state["current_player"]
                self.state["result"] = "move accepted"
        else:
            self.state["result"] = "move invalid"
            
    def get_valid_moves(self):
        return TicTacToe.get_valid_moves_from_state(self.state["board"])

    def get_board_size(self):
        return self.board_size

    def print_board(self):
        print("-------------")
        for i in range(3):
            row = [str(cell).replace('1', 'X').replace('-X', 'O').replace('0', ' ') for cell in self.state["board"][i*3:(i*3)+3]]
            print(f"| {row[0]} | {row[1]} | {row[2]} |")
            if i < 2:
                print("-------------")
        print("-------------")

    @staticmethod
    def get_valid_moves_from_state(state_board):
        """Retorna una lista de índices de movimientos válidos (0-8) a partir de un estado de tablero."""
        # Se asume que state_board es la tupla/lista de 9 elementos
        return [i for i in range(9) if state_board[i] == 0]

    @staticmethod
    def state_to_array(board):
        """Convierte el tablero (lista de 9) en un array de numpy (1, 9)."""
        return np.array(board).reshape(1, -1)

