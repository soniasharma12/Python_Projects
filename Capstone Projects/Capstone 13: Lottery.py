# Create a list containing the number of favourable outcomes per draw for random variables X = 0 to X = 10.
fav_out = [[10 - j if j < i else 60 + i - j for j in range(10)] for i in range(11)]
fav_out

# Create a list containing probabilities of the random variable X = 0 to X = 10.
all_outcomes=[]
sum=0
for i in range(11):
  all_outcomes.append(ncr(10,i))
  sum=sum+ncr(10,i)
  
prob_list=[]
prob_sum=0
for i in range(11):
  prob_list.append(all_outcomes[i]/sum)
  prob_sum+=(all_outcomes[i]/sum)
print(prob_list)
prob_sum

# Function to calculate the factorial value of a number 'n'.
def factorial(num):
  fact=1
  if num<0:
    return "UnDefined"
  else:
    while num>0:
      fact*=num
      num-=1
  return fact
print(f"Factorial of 20 = {factorial(20)}")

win_lst=[-250,-250,-250,-250,-250,250,2500,25000,250000,2500000,25000000]
expected_winning=0
for i in range(11):
  expected_winning+= win_lst[i]*prob_list[i]

print(f"{expected_winning:.2f}")
