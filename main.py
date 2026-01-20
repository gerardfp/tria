# Archivo: main.py

from tic_tac_toe import TicTacToe
from nim import Nim
from player import Player
from dqn_ia import DQN_IA 

GAME_CLASS = Nim
AGENT_CLASS = DQN_IA

def train_ia(GameClass, AgentClass, episodes):    
    ia_player1 = AgentClass(GameClass=GameClass, debug=False)
    ia_player2 = AgentClass(GameClass=GameClass) 
    
    print(f"Iniciando entrenamiento de {AGENT_CLASS.__name__} en {GameClass.__name__} para {episodes} episodios.")
    
    gamma = 0.9 # Factor de descuento para la propagación de recompensa
    
    for episode in range(episodes):
        game = GameClass(ia_player1, ia_player2, quiet=True)
        
        history_p1 = [] 
        history_p2 = []
        
        while game.winner is None:
            player_who_moved = game.current_player
            old_board = tuple(game.board)
            
            move = game.next() 
            if move is None: 
                 break 
            
            new_board = tuple(game.board)
            
            transition = {
                "state": old_board,
                "action": move,
                "reward": 0.0,
                "next_state": new_board,
            }
            
            if player_who_moved is ia_player1:
                history_p1.append(transition)
            else:
                history_p2.append(transition)
            
        if game.winner is not None:
            if game.winner is ia_player1:
                reward_p1, reward_p2 = 1.0, -1.0
            elif game.winner is ia_player2:
                reward_p1, reward_p2 = -1.0, 1.0
            else: # Empate
                reward_p1, reward_p2 = 0.1, 0.1
            
            for i, trans in enumerate(reversed(history_p1)):
                trans["reward"] = reward_p1 * (gamma ** i)
            
            for i, trans in enumerate(reversed(history_p2)):
                trans["reward"] = reward_p2 * (gamma ** i)

            ia_player1.learn(history_p1)
            ia_player2.learn(history_p2)
                
        if (episode + 1) % (episodes // 10) == 0:
             print(f"Episodio {episode + 1}/{episodes} completado.")

    print(f"Entrenamiento terminado.")
    return ia_player1 

def main():    
    print(f"INICIANDO {GAME_CLASS.__name__}")
    p1 = train_ia(GAME_CLASS, AGENT_CLASS, episodes=1000)
    
    p1.epsilon = 0.0
    p2 = Player()
    p1.debug = True

    for i in range(10):
        game = GAME_CLASS(p1, p2, quiet=False) 
        print(f"\n--- 🎭🎭🎭 ¡COMIENZA EL JUEGO {p1} vs {p2}!")

        while game.winner is None:
            game.print_board()
            game.next()

        game.print_board()
        if game.winner == "draw":
            print("--- 🤝 ¡EMPATE! ---")
        else:
            print(f"--- 🏆 ¡GANADOR: {game.winner}! ---")

if __name__ == "__main__":
    main()