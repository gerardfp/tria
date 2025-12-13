class Player:
    """Representa un jugador humano, ajustado para el Nim."""
    def __init__(self, mark=-1):
        self.mark = mark

    def __repr__(self):
        return "Human"
    
    def make_move(self, state):
        return int(input())-1