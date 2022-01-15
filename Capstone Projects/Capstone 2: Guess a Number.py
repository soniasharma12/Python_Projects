# Create the 'get_binary_function()'.
def get_binary_digits(dividend):
  binary_digits=[]
  while dividend != 0:
    quot = dividend // 2
    remainder = dividend % 2
    binary_digits.append(remainder)
    dividend = quot
  binary_digits.reverse()
  return binary_digits
get_binary_digits(53)

# Create an empty list called 'cards' containing 6 other empty lists.
cards=[[] for i in range(1,7)]
print(cards)

# Fill the 'cards' list by applying the game logic in python.
first_six_powers_of_two = [2**i for i in range(6)]
for num in range(1,64):
  bin_digits = get_binary_digits(num)
  for i in range(len(bin_digits)):
    power_of_two = 2**((len(bin_digits)-1)-i) 
    if bin_digits[i]*power_of_two in first_six_powers_of_two:
      cards[((len(bin_digits)-1)-i)].append(num)
      
# Run the game here.
player_input = input("Think of a Number Betwen 1-63. Type start when you're ready and hit enter to start!:\n ")
while player_input != 'start':
  player_input = input("Think of a Number Betwen 1-63. Type start when you're ready and hit enter to start!:\n ")

num = 0
valid_enteries = ["Yes", "No"]
for i in range(len(cards)):
  print("Does you're number exist on card?", i + 1, "Card", i + 1, "==>", cards[i])
  user_input = input("Does your number exist? Enter Yes or No: ")
  while user_input not in valid_enteries:
    print("Invaild Entry: ")
    print("Does you're number exist on card?", i + 1, "Card", i + 1, "==>", cards[i])
    user_input = input("Does your number exist? Enter Yes or No: ")
  if user_input == "Yes": 
    num = num + cards[i][0]
print("You thought of the number: ", num)

