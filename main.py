# Archivo: main.py

from tic_tac_toe import TicTacToe
from nim_game import NimGame
from player import Player
from dqn_ia import DQN_IA 
from q_learning_ia import QLearningIA


GAME_CLASS = NimGame
AGENT_CLASS = DQN_IA

def train_ia(GameClass, AgentClass, episodes):
    """Función genérica para entrenar cualquier agente en cualquier juego."""
    
    ia_player1 = AgentClass(mark=1, GameClass=GameClass)
    ia_player2 = AgentClass(mark=-1, GameClass=GameClass) 
    
    print(f"Iniciando entrenamiento de {AGENT_CLASS.__name__} en {GameClass.__name__} para {episodes} episodios.")
    
    for episode in range(episodes):
        game = GameClass(ia_player1, ia_player2, quiet=True)
        
        # Historial de transiciones en la partida para cada jugador: (s, a, r, s')
        history_p1 = [] 
        history_p2 = []
        
        while game.state["winner"] is None:
            current_player = game.state["current_player"]
            
            old_board_key = tuple(game.state["board"]) 
            
            move = game.next() 
            
            if move is None: 
                 break 
            
            new_board_key = tuple(game.state["board"])
            
            if current_player is ia_player1:
                history_p1.append([old_board_key, move, 0.0, new_board_key])
            else:
                history_p2.append([old_board_key, move, 0.0, new_board_key])
            
            
            if game.state["winner"] is not None:
                reward_p1 = 0.0
                reward_p2 = 0.0
                
                if game.state["winner"] is ia_player1:
                    reward_p1, reward_p2 = 1.0, -1.0
                elif game.state["winner"] is ia_player2:
                    reward_p1, reward_p2 = -1.0, 1.0
                elif game.state["winner"] == "draw":
                    reward_p1, reward_p2 = 0.1, 0.1
                
                if history_p1:
                    history_p1[-1][2] = reward_p1
                if history_p2:
                    history_p2[-1][2] = reward_p2

                ia_player1.learn(history_p1)
                ia_player2.learn(history_p2)

                break
                
        if (episode + 1) % (episodes // 100) == 0:
             print(f"Episodio {episode + 1}/{episodes} completado.")

    print(f"Entrenamiento terminado.")
    return ia_player1 

def main():
    episodes = 20000 
    
    print(f"\n--- INICIANDO {GAME_CLASS.__name__.upper()} ---")
    trained_ia = train_ia(GAME_CLASS, AGENT_CLASS, episodes)
    
    trained_ia.epsilon = 0.0

    human_player = Player(mark=-1)
    
    print(f"\n--- ¡COMIENZA EL JUEGO CONTRA EL HUMANO! ---")
    game = GAME_CLASS(trained_ia, human_player) 
    
    while game.state["winner"] is None:
        game.print_board()
        game.next() 
        
        if game.state["winner"] is not None:
            game.print_board()
            winner_info = game.state['winner']
            if winner_info == 'draw':
                 print("¡Juego terminado! Resultado: Empate")
            else:
                 print(f"¡Juego terminado! Ganador es: {winner_info} (Marca: {'X' if winner_info.mark == 1 else 'O'})")
            break

if __name__ == "__main__":
    main()