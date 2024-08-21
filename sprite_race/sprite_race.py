import sprite_race_functions as sr
from collections import deque, Counter, namedtuple
from time import time, sleep

maze_file_name = 'maze_data_2.csv'
seconds_between_turns = 0.3
max_turns = 35

#initialize the sprite race
maze_data = sr.read_maze(maze_file_name)
sr.print_maze(maze_data)
walls, goal, bots = sr.process_maze_init(maze_data)

#populate a deque of all sprite commands for the provided maze
sprite_moves = deque()
num_of_turns = 0
while not sr.is_race_over(bots) and num_of_turns < max_turns:
  #for every bot in the list of bots, if the bot has not reached the end, add a new move to the sprite_moves deque
  for bot in bots:
    if bot.has_finished == False:
      sprite_moves.append(sr.compute_sprite_logic(walls, goal, bot))

  num_of_turns += 1

#count the number of moves based on the sprite names
move_count = Counter(move[0] for move in sprite_moves)

#count the number of collisions by sprite name
collision_count = Counter(move[0] for move in sprite_moves if move[2] == True)

#create a namedtuple to keep track of our sprites' points
BotScoreData = namedtuple('BotScoreData', 'name, num_moves, num_collisions, score')

#calculate the scores (moves + collisions) for each sprite and append it to bot_scores 
bot_scores = []
for bot in bots:
  nm = bot.name
  bot_scores.append(BotScoreData(nm, move_count[nm], collision_count[nm], move_count[nm] + collision_count[nm]))

#populate a dict to keep track of the sprite movements
bot_data = {}
for bot in bots:
  bot_data[bot.name] = bot

#move the sprites and update the map based on the moves deque
while len(sprite_moves) > 0:
  #pop moves from the front of the deque
  bot_name, direction, has_collided = sprite_moves.popleft()
  bot_data[bot_name].process_move(direction)

  #update the maze characters based on the sprite positions and print it to the console
  sr.update_maze_characters(maze_data, bots)
  sr.print_maze(maze_data)
  sleep(seconds_between_turns - time() % seconds_between_turns)

#print out the results
sr.print_results(bot_scores)