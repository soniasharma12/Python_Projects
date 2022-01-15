#  Create class 'Employee'. Add constructor to initialise variables. Add getter and setter functions.
class Employee:
  def __init__(self, empid, empname, ssn):
    self.empid = empid
    self.empname = empname
    self.ssn = ssn
    self.income = None
  
  def get_empid(self):
    return self.empid
  def set_empid(self, new_empid):
    self.empid = new_empid
  
  def get_empname(self):
    return self.empname
  def set_empname(self, new_empname):
    self.empname = new_empname

  def get_ssn(self):
    return self.ssn
  def set_ssn(self, new_ssn):
    self.ssn = new_ssn
  
  def get_income(self):
    return self.income
  
  def calc_income(self):
    pass
  
  def __repr__(self):
    return f" Employee ID: {self.get_empid()}\n Employee Name: {self.get_empname()}\n Social Security #: {self.get_ssn()}\n Calculated Income: {self.calc_income()}"
# Object
emp_1 = Employee(1224, 'Sonia Sharma', 57)
print(emp_1)

#  Create a child class 'SalariedEmployee'. Add constructor and override 'calculate_income()' function.
class SalariedEmployee(Employee):
  def __init__(self,empid,empname,ssn):
    Employee.__init__(self,empid,empname,ssn)
    self.__ta = 100
    self.__da = 200
    self.__hra = 300

  def get_ta(self):
    return self.__ta

  def get_da(self):
    return self.__da

  def get_hra(self):
    return self.__hra

  def set_ta(self,new_ta):
    self.__ta = new_ta

  def set_da(self,new_da):
    self.__da = new_da

  def set_hra(self,new_hra):
    self.__hra = new_hra

  def  calculated_income(self):
    basic = int(input("Enter employee basic salary : "))
    self.income = basic + self.__ta + self.__da + self.__hra
    return self.income 
    
  def __repr__(self):
    return f"Employee ID : {self.get_empid()}: \nEmployee Name : {self.get_empname()} \nSocial Security #: {self.get_ssn()}  \nTravelling Allowance : {self.get_ta()} \nDearness Allowance : {self.get_da()} \nHouse Rent Allowance : {self.get_hra()} \nCalculated income : {self.calculated_income()}"
# Object
obj2 = SalariedEmployee(1224,"Sonia",21)
print(obj2)

#  Create a child class 'HourlyEmployee'. Add constructor and override 'calculate_income()' function
class HourlyEmployee(Employee):
  def __init__(self,empid,empname,ssn):
    Employee.__init__(self,empid,empname,ssn)
    self.__per_hour = 150

  def get_perhour(self):
    return self.__per_hour

  def set_perhour(self,new_hour):
    self.__per_hour = new_hour

  def calculate_income(self):
    time = int(input("Enter number of hours an employee worked "))
    self.income = time * self.__per_hour
    return self.income

  def __repr__(self):
    return f"Employee ID : {self.get_empid()} \nEmployee Name : {self.get_empname()} \nSocial Security #: {self.get_ssn()} \nCalculated income : {self.calculate_income()} \nWage paid per hour : {self.get_perhour()}"  
obj3 = HourlyEmployee(212412,"Sonia Sharma",12)
print(obj3)

#  Create a child class 'CommissionEmployee'. Add constructor and override 'calculate_income()' function.
class CommissionEmployee(Employee):
  def __init__(self,empid,empname,ssn):
    Employee.__init__(self,empid,empname,ssn)
    self.__commission_rate = 0.2

  def get_commission(self):
    return self.__commission_rate

  def set_commission(self,new_comission):
    self.__commission_rate = new_comission

  def calculate_income(self):
    gross_sales = int(input("Enter gross sales "))
    self.income = gross_sales * self.__commission_rate
    return self.income

  def __repr__(self):
    return f"Employee ID : {self.get_empid()} \nEmployee Name : {self.get_empname()} \nSocial Security Number : {self.get_ssn()}  \nComission Rate : {self.get_commission()} \nCalculated income : {self.calculate_income() }  "
# Object
obj4 = CommissionEmployee(3456,"Sonia Sharma",12)
print(obj4)

# Create class 'Management'.
class Management:
  emp_id_list = []
  emp_records = []
  # Create 'existing_employee' function
  def existing_employee(emp_id):
    if emp_id in Management.emp_id_list:
      print("This employee ID already exists")
      return True
    else :
      Management.emp_id_list.append(emp_id)
  # Create class method 'add_records()'
  @classmethod
  def add_records(cls,employee):
    dict1 = {}
    dict1["Employee Id"] = employee.get_empid()
    dict1["Employee Name"] = employee.get_empname()
    dict1["Social Security Number"] = employee.get_ssn()
    dict1["Employee Income"] = employee.get_income()
    Management.emp_records.append(dict1)
  # Create class method 'display_emp()' 
  @classmethod
  def display_records(cls):
    return Management.emp_records
# Object
manage = Management()
manage.add_records(obj4)
manage.display_records()

# Create  infinite while loop.
while True:
  user_empid = int(input("Enter Employee Id : "))
  while  Management.existing_employee(user_empid) == True:
    print("This Id already exists")
    print("Enter a fresh Id")
    user_empid = imt(input("Enter user Id : "))

  user_empname = input("Enter employee  Name : ")
  user_ssn = int(input("Enter Social Security Number : "))
  
  valid_inputs = [1,2,3]
  user_input = int(input(f"\nEnter 1 for Salaried employee \n Enter 2 for Hourly Employee \n Enter 3 for Comission Employee \n"))
  while user_input not in valid_inputs:
    print("\nInvalid Choice")
    uer_input = int(input("Enter a valid choice : "))

  if user_input == 1:
    employee = SalariedEmployee(user_empid,user_empname,user_ssn)
    
    valid_choice = [1,2,3,4]
    while True:
      user_choice = int(input(f"\nEnter 1 for updating Travelling allowance(TA)  \nEnter 2 for updating Dearness allowance(DA) \nEnter 3 for updating House Rent allowance(HRA) \nEnter 4 to skip updating \n"))
       
      while user_choice not in valid_choice:
        print("\nPlease Enter A valid Input")
        user_choice = int(input(f"\nEnter 1 for updating Travelling allowance(TA)  \nEnter 2 for updating Dearness allowance(DA) \nEnter 3 for updating House Rent allowance(HRA) \nEnter 4 to skip updating \n"))
      
      if user_choice == 1:
        employee.set_ta(int(input("\nEnter Travelling Allowance (TA) : ")))
        print("\nTravelling Allowance Updated")

      elif user_choice == 2:
        employee.set_da(int(input("\nEnter Dearness Allowance (DA) : ")))
        print("\nDearness Allowance Updated")  

      elif user_choice == 3:
        employee.set_hra(int(input("\nEnter House Rent Allowance (HRA) : ")))
        print("\nHouse Rent Allowance Updated")
      
      else:
        break
  
  elif user_input == 2:
    employee = HourlyEmployee(user_empid,user_empname,user_ssn)
    user_entry = int(input("\nEnter 1 for updating hourly wages \nEnter 2 to skip updating \n"))

    valid_entry = [1,2]
    while user_entry not in valid_entry:
      print("\nInvalid choice ")
      user_entry = int(input("\nEnter a valid choice "))

    if user_entry == 1:
      employee.set_perhour(int(input("\nEnter wages per hour")))
      print("\nWages per hour updates")

    elif user_entry == 2:
      break

  else:
    employee = CommissionEmployee(user_empid,user_empname,user_ssn)
    users_entry = int(input("\nEnter 1 for updating Commission \nEnter 2 to skip updating \n"))        

    while users_entry not in [1,2]:
      print("\nInvalid choice ")
      users_entry = int(input("\nEnter a valid choice \n"))

    if users_entry == 1:
      employee.set_commission(int(input("\nEnter Comission Rate : ")))
      print("Commission rate updated")

    else:
      break

  user_wish = int(input("Enter 1 to continue updation \nEnter 2 to quit \n"))        
       
  while user_wish not in [1,2]:
    print("\nInvalid choice")
    user_wish = int(input("Enter a valid choice \n"))     

  if user_wish == 1:
    print(employee)
    print(f"{employee.calculate_income}")
    Management.add_records(employee)  

    display_records = int(input('\nEnter 1 to view the records of all employees \nEnter 2 to skip \n'))
    while display_records not in [1, 2]:
      print( "INVALID INPUT")
      display_records = int(input("Enter a valid choice \n"))
    
    if display_records == 1:
      print(Management.display_records())
      
    else:
      break

    run = int(input('\nEnter 1 to continue running the application \nEnter 2 to quit \n'))
    while run not in [1, 2]:
      print('\nInvalid input')
      display_records = int(input('Enter a valid choice /n'))
    
    if run == 1:
      print(Management.display_records())
      break

    else:
      break
      
