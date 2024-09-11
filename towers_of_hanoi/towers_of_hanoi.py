from stack import Stack

#opening screen
def opening():
  print('''
  
                                  o
                              .-'"|
                              |-'"|
                                  |   _.-'`.
                                 _|-"'_.-'|.`.
                                |:^.-'_.-'`.;.`.
                                | `.'.   ,-'_.-'|
                                |   + '-'.-'   J
             __.            .d88|    `.-'      |
        _.--'_..`.    .d88888888|     |       J'b.
     +:" ,--'_.|`.`.d88888888888|-.   |    _-.|888b.
     | \ \-'_.--'_.-+888888888+'  _>F F +:'   `88888bo.
      L \ +'_.--'   |88888+"'  _.' J J J  `.    +8888888b.
      |  `+'        |8+"'  _.-'    | | |    +    `+8888888._-'.
    .d8L  L         J  _.-'        | | |     `.    `+888+^'.-|.`.
   d888|  |         J-'            F F F       `.  _.-"_.-'_.+.`.`.
  d88888L  L     _.  L            J J J          `|. +'_.-'    `_+ `;
  888888J  |  +-'  \ L         _.-+.|.+.          F `.`.     .-'_.-"J
  8888888|  L L\    \|     _.-'     '   `.       J    `.`.,-'.-"    |
  8888888PL | | \    `._.-'               `.     |      `..-"      J.b
  8888888 |  L L `.    \     _.-+.          `.   L+`.     |        F88b
  8888888  L | |   \   _..--'_.-|.`.          >-'    `., J        |8888b
  8888888  |  L L   +:" _.--'_.-'.`.`.    _.-'     .-' | |       JY88888b
  8888888   L | |   J \ \_.-'     `.`.`.-'     _.-'   J J        F Y88888b
  Y888888    \ L L   L \ `.      _.-'_.-+  _.-'       | |       |   Y88888b
  `888888b    \| |   |  `. \ _.-'_.-'   |-'          J J       J     Y88888b
   Y888888     +'\   J    \ '_.-'       F    ,-T"\   | |    .-'      )888888
    Y88888b.      \   L    +'          J    /  | J  J J  .-'        .d888888
     Y888888b      \  |    |           |    F  '.|.-'+|-'         .d88888888
      Y888888b      \ J    |           F   J    -.              .od88888888P
       Y888888b      \ L   |          J    | .' ` \d8888888888888888888888P
        Y888888b      \|   |          |  .-'`.  `\ `.88888888888888888888P
         Y888888b.     J   |          F-'     \\ ` \ \88888888888888888P'
          Y8888888b     L  |         J       d8`.`\  \`.8888888888888P'
           Y8888888b    |  |        .+      d8888\  ` .'  `Y888888P'
           `88888888b   J  |     .-'     .od888888\.-'
            Y88888888b   \ |  .-'     d888888888P'
            `888888888b   \|-'       d888888888P
             `Y88888888b            d8888888P'
               Y88888888bo.      .od88888888
               `8888888888888888888888888888
                Y88888888888888888888888888P
                 `Y8888888888888888888888P'
                   `Y8888888888888P'
                        `Y88888P' ''')
  print("\n                             ENTER THE TOWERS OF HANOI")

#get user to choose a move
def get_input(stacks):
  choices = [stack.get_name()[0] for stack in stacks]
  while True:
    for i in range(len(stacks)):
      name = stacks[i].get_name()
      letter = choices[i]
      print('Enter {letter} for {name}'.format(letter=letter, name=name))
    
    user_input = input('')

    if user_input in choices:
      for i in range(len(stacks)):
        if user_input == choices[i]:
          return stacks[i]

#play the game
def gameplay():
  #create the stacks
  stacks = []
  left_stack, middle_stack, right_stack = Stack('Left'), Stack('Middle'), Stack('Right')
  stacks.append(left_stack)
  stacks.append(middle_stack)
  stacks.append(right_stack)

  #choose number of disks to play with
  num_disks = int(input('\nHow many disks do you want to play with?\n'))

  while (num_disks < 3):
    num_disks = int(input('Enter a number greater than or equal to 3\n'))

  for i in range(num_disks, 0, -1):
    left_stack.push(i)

  #print optimal number of moves
  num_optimal_moves = (2 ** num_disks) - 1
  print('\nThe fastest you can solve this game is in {} moves'.format(num_optimal_moves))

  #gameplay loop
  num_user_moves = 0
  while (right_stack.get_size() != num_disks):
    #print our current stacks
    print('\n\n\n...Current Stacks...')
    for s in stacks:
      s.print_items()

    while True:
      print('\nWhich stack do you want to move from?\n')
      from_stack = get_input(stacks)
      print('\nWhich stack do you want to move to?\n')
      to_stack = get_input(stacks)

      if from_stack.is_empty():
        print('Invalid Move. Try Again.')
      elif (to_stack.is_empty()) or (from_stack.peek() < to_stack.peek()):
        disk = from_stack.pop()
        to_stack.push(disk)
        num_user_moves += 1
        break
      else:
        print('\n\nInvalid Move. Try Again.')

  print('\n\nYou completed the game in {user_moves} moves, and the optimal number of moves is {optimal_moves}'.format(user_moves=num_user_moves, optimal_moves=num_optimal_moves))

#run the file
def main():
  #display opening
  opening()
  
  #keep playing if user wants to
  keep_playing = True
  while keep_playing:
    gameplay()
    user_input = input("\n                             PRESS Y TO PLAY AGAIN\n")
    
    keep_playing = False
    if user_input.lower() == 'y':
      keep_playing = True

#actually execute the file
try:
  main()
except:
  print("Error loading towers_of_hanoi.py")