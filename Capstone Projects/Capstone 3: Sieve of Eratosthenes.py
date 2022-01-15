n = int(input("Please enter a number till which you need all primes: "))
primes = [i for i in range(n + 1)]

print(primes)
for prime in range(2, len(primes)):
  current_prime = prime
  if current_prime != 0: 
    # Write your code from here.  
    for i in range(current_prime * 2, len(primes), current_prime):
      primes[i] = 0

print(primes)
primes = primes[2:]

print(primes)

