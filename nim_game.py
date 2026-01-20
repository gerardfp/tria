import numpy as np

class NimGame():
    """Implementa las reglas del juego 'Ataque del 10'.
    El que resta el último número gana.
    """
    # Constantes del juego
    
    MAX_NIM = 7
    ACTIONS = [1, 2, 3] # Movimientos permitidos: restar 1, 2, o 3

    CONFIG = {
        "input_size": MAX_NIM + 1, 
        "output_size": len(ACTIONS),
        "hidden_size": 64,      
        "game_actions": ACTIONS,
    }

    def __init__(self, p1, p2, quiet=False):
        self.initial_nim = self.MAX_NIM
        p1.mark = 1
        p2.mark = -1
        self.state = {
            "board": [self.initial_nim],
            "current_player": p1,
            "other_player": p2,
            "winner": None,
            "valid_moves": self.ACTIONS,
        }
        self.actions = self.ACTIONS
        self.quiet = quiet

    def next(self):
        remaining = self.state["board"][0]
        while True:
            try:
                if not self.quiet:
                    print(f"\n--- Turno {self.state['current_player']} (Quedan: {remaining}) ---")
                    print(f"Puedes restar: {[move for move in self.ACTIONS if move <= remaining]}")
                i = int(self.state["current_player"].make_move(self.state))
                
                if 0 <= i < len(self.actions) and  self.actions[i] <= remaining:
                    move = self.actions[i] 
                    self.process(move) 
                    return i
                elif not self.quiet:

                    print(f"Movimiento inválido. Debes restar {[move for move in self.ACTIONS if move <= remaining]} y no exceder {remaining}.")
            except ValueError:
                if not self.quiet:
                    print("Entrada inválida. Usa {[move for move in self.ACTIONS if move <= remaining]}.")
        

    def _win(self):
        if self.state["board"][0] == 0:
            return 1 
        return 0 
    
    def process(self, cmd):
        move_value = cmd
        remaining = self.state["board"][0]

        if move_value in self.actions and move_value <= remaining:
            self.state["board"][0] -= move_value
            
            win = self._win()
            
            if win != 0:
                self.state["winner"] = self.state["current_player"]
            else:
                self.state["current_player"], self.state["other_player"] = self.state["other_player"], self.state["current_player"]
                self.state["result"] = "move accepted"
        else:
            self.state["result"] = "move invalid" 
            
    def get_valid_moves(self):
        return NimGame.get_valid_moves_from_state(self.state["board"])

    def get_board_size(self):
        return self.MAX_NIM + 1

    def print_board(self):
        print(f"Número restante: {self.state['board'][0]}")

    @staticmethod
    def get_valid_moves_from_state(state_board):
        remaining = state_board[0]
        return [i for i, move in enumerate(NimGame.ACTIONS) if move <= remaining]

    @staticmethod
    def state_to_array(board):
        """Convierte el tablero (un solo número) en un vector One-Hot."""
        remaining = board[0]
        arr = np.zeros((1, NimGame.MAX_NIM + 1))
        arr[0, remaining] = 1.0
        return arr