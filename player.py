class Player:
    """Representa un jugador humano, ajustado para el Nim."""
    def __init__(self):
        self.mark = None


    def __repr__(self):
        return "Human"
    
    def make_move(self, state):
        return int(input())-1