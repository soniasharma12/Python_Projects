# Create the 'cust_data' dictionary and the 'new_user_attributes' list.
cust_data = {}
new_user_attributes = ['name', 'address', 'phone num', 'govt id', 'amount']

# Create the 'new_user()' function to add a new user to the 'cust_data' dictionary. 
from random import randint

def new_user():

  acc_no = str(randint(10000, 99999))
  new_user_input = []
  
  for i in range(len(new_user_attributes)):
    new_user_input.append(input(f"enter your {new_user_attributes[i]} : "))

  new_user_details = {k : v for k,v in zip(new_user_attributes, new_user_input)}
  cust_data[acc_no] = new_user_details

  print(f"\nYour details are added successfully.\
        \nYour account number is {acc_no}\
        \nPlease don't lose it.")
  
# Create the 'existing_user()' function to get the account details of an existing user from the 'cust_data' dictionary.
def existing_user():

  vaild_option = ["1", "2", "3"]

  user_acc = input("Please enter your correct account number:")

  while user_acc not in cust_data.keys():
    print(f"Not found.\n{user_acc}")
  
  details = cust_data[user_acc]

  available_amount = details['amount']
  available_amount = int(available_amount)

  user_input = input(f"Welcome, {details['name']} !\
                          \nEnter 1 to check your balance.\
                          \nEnter 2 to withdraw an amount.\
                          \nEnter 3 to deposit an amount.")

  while user_input not in vaild_option:
    print(f"Invalid input!\n{user_input}")

  if user_input == "1":
    print(f"\nyour account balance is {available_amount}")

  elif user_input == "2":
    w_amount = int(input("Enter the amount to be withdrawn:"))

    if w_amount > available_amount:
      print(f"Insufficient balance.\nAvailable balance: {available_amount}")

    else:
      available_amount -= w_amount
      print(f"Withdrawal successful.\nAvailable Balance: {available_amount}")

  elif user_input == "3":
    d_amount = int(input("Enter the to be deposite"))
    available_amount += d_amount
    print(f"Deposit successful.\nAvailable Balance: {available_amount}")
    
# Create an infinite while loop to run the application.
def app():

  last_lines = "\nThank you, for banking with us!"

  while True:
    valid_inputs = ['1', '2', '3']

    print("Welcome to the Horizon Bank!")
    print()

    option = input("\nEnter 1 if you are a new customer.\
                    \nEnter 2 if you are an existing customer.\
                    \nEnter 3 to terminate the application.\n")

    while option not in valid_inputs:
      print("\nInvalid input!")
      print(option)
    if option == "1":
      new_user()
      print(last_lines)
      break

    elif option == "2":
      existing_user()
      print(last_lines)
      break

    elif option == "3":
      print(last_lines)
      break

app()
