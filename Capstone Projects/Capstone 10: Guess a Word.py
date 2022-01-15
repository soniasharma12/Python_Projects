# Create 'my_words' & 'records' tuples.
my_words = (("questionnaire", "noun", "a list of questions survey"),
            ("unconscious", "adjective", "not conscious or without awareness"),
            ("precocious", "adjective", "unusally mature, especially in mental development"),
            ("liasion", "noun", "a person who maintains a connection between people or groups"),
            ("surveillance", "noun", "continous observation of a person, place, or activity in order to gather information"),
            ("malfeasance", "noun", "conduct by public officials that violates the public trust or is against the law"),
            ("irasacible", "adjective", "irritable, quick-tempered"),
            ("idiosyncrasy", "noun", "a tendency, habit or mannerism that is peculiar to an individual; aquirk"),
            ("foudroyant", "adjective", "sudden and overwhelming or stunning"),
            ("eudemonic", "adjective", "pertaining to conducive to happiness"))
records = ({"name" : [], "guess count" : [], "time taken" : []}, )
d = records[0]

# Create the 'shuffler()' function.
import random
def shuffler(word):
  """This fuction returns the jumbled word"""
  letter_list = list(word)
  random.shuffle(letter_list)
  jumbled_letters = "".join(letter_list)
  return jumbled_letters

# Remaining Components 
import time
title = "GUESS A WORD GAME".center(120, "-")
print(title, "\n")

name = input("Enter your name.\nEnter 's' as an input to stop the game midway.\n")
guess_count = 0
start = time.time()

for i in my_words:
  print("\nJumbled Letters:", shuffler(i[0]))
  print("Part of speech:", i[1])
  print("Meaning:", i[2])

  guess = input("\nGuess the word :\n").lower()

  if guess == "s":
    break

  elif guess == i[0]:
    print("\nCORRECT!")
    guess_count += 1
    print("your score is", guess_count)

  elif guess != i[0]:
    print("\nWRONG! \nCorrect word is", i[0])

stop = time.time()
time_taken = stop - start

if time_taken < 60:
  print("\nYou took {:.0f} second to guess {} words correctly.".format(time_taken, guess_count))
else:
  print("\nYou took {:.0f} minute(s) and {} seconds to guess {} words correctly.".format(time_taken//60, str(time_taken / 60 - 60)[:2], guess_count))

print("\nComplete Word List")
print("-"*120)
print("##|".ljust(3, "|") + "Word|".rjust(len("questionnaire")+6, " ") + "Parts of Speech|".rjust(len("Parts of Speech|")+6, " "), "Meaning")
print("-"*120)

for i in range(len(my_words)):
  s_no = "{:02}".format(i+1).ljust(3, "|")
  word = my_words[i][0].rjust(len("questionnaire")+5, " ")
  p_speech = my_words[i][1].rjust(len("Parts of Speech|")+5, " ")
  meaning = my_words[i][2]
  
  print(s_no + word.ljust(len(word) + 1, "|") + p_speech.ljust(len(p_speech) + 1, "|"), meaning.capitalize())
  print("-"*120)

d["name"].append(name)
d["guess count"].append(guess_count)
d["time taken"].append(time_taken)

print("\nLEADERBOARD")
print("-"*120)
for i in range(len(d["name"])):
  print("\nName :", d["name"][i])
  print("Guess Count :", d["guess count"][i])
  print("Time Taken :", d["time taken"][i]) 
