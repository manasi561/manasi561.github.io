---
layout: post
title: Guessing the number
image: "/posts/number.jpg"
tags: [Python]
---

In this game, the program generates a random number between 1 and 100, which the user has to guess, but this number is not visible to the user. The user tries to guess the number in finite number of chances. 

If the user correctly guesses the number, the game ends and a winning message is displayed. 

If the user enters a wrong number then that number is compared with the right answer. If the number is greater than right answer then the program gives a hint that entered number is ‘too high’ else if number is smaller than right answer than it says ‘too low’ and also displays number of guess remainng. We also display a warning message if the user guesses a number out of bound.

Let's get into it!

We import random module which is used to generate random numbers


```python
import random 
```

The randint(*start*, *stop*) method returns an integer from the specified range. In this case, it will return a random integer between 1 and 100. Our game will generate a random number between 1 and 100, which the user has to guess. We will store this number in a variable, correct_number.


```python
correct_number = random.randint(1, 100)
```

Let's store the lower_bound and upper_bound values as well as the number of guesses used and the number of guesses remaining. As we mentioned earlier while generating random numbers that the range of numbers is between 1 and 100, we will have a lower_bound of 1 and an upper_bound of 100. This is the bound between which the user has to guess. 

Let us give user 5 chances to guess the correct answer. This is the maximum number of chances user will have to guess the right answer.

The guess_counter variable will be used to keep a track of the number of guesses user has utilized.  


```python
lower_bound = 1
upper_bound = 100
guess_remaining = 5
guess_counter = 0
```

We now have all the required data with us, next, we move to write logic, which will check if the guess entered by the user is correct or not and display a relevant message.

This message is displayed to the user before they start of the game so that they are equipped with all the requied information before they begin.


```python
print(f"Guess the num between {lower_bound} and {upper_bound}. You have {guess_remaining} chances")
```

While loop is used as it will keep the process running until a certain condition is met. For the scope of this game, we want the loop to run, till the user has guessed the correct answer or if they have exhausted their number of guesses.

'While True' which essentially tells python to keep running loop until it encounters break clause within the loop. 
The events that will trigger the break will be:
    i) The user guesses the number correctly.
    ii) The user has exhausted their guesses.
    
We use the print statement before the break clause to notify user the reason of terminating the game.

Once the user has all the required information, we ask the user to enter their guess. We use the int(input()) function to ensure the user only enters integer.

As the user enters their guess, we will increment the number of guesses the user has had so far by increasing the guess_counter. Simultaneously, we also deduce the reamining number of guesses the user has as they have utilized one of their guesses.

We then introduce some conditional logic which will come into operartion if the user guess is within the upper_bound and lower_bound.
    
  If the user guesses the number correctly, we display a congratulatory message and also, notify the user the number of attempts   in which they got their guess right, before breaking the loop.
  
  If their guess is incorrect, we include a logic which will notify the user if their attempt is too high or too low.
  We also display a message which will inform them about the remaining guesses.
  
If the user guesses a value beyond the bounds set, we display an alert messaging indicating that their guess is outside of the range. 

Lastly, when the user has exhausted all their guesses, we will break the before with a message informing them that they have run out of guesses and also about the correct number.
  


```python

while True:
    user_guess = int(input("Enter your guess: "))
    guess_counter += 1    
    guess_remaining = guess_remaining-1
    
    if lower_bound <= user_guess <= upper_bound:
        if user_guess == correct_number:
            print(f"Congrats! You got it in {guess_counter} guess")
            break
        
        elif user_guess < correct_number:
            print(f"Your guess is too low, try again! Guess remaining: {guess_remaining}")
        
        else:
            print(f"Your guess is too high, try again! Guess remaining: {guess_remaining}")
            
            
    else:
        print(f"Your guess is out of range. Please select a number between {lower_bound} and {upper_bound}. Guess remaining: {guess_remaining}")
        
        
    if guess_remaining == 0:
        print(f"Sorry you are out of guesses. Correct number is {correct_number}")
        break
```

    Guess the num between 1 and 100. You have 5 chances
    Enter your guess: 50
    Your guess is too low, try again! Guess remaining: 4
    Enter your guess: 75
    Your guess is too low, try again! Guess remaining: 3
    Enter your guess: 90
    Your guess is too high, try again! Guess remaining: 2
    Enter your guess: 85
    Your guess is too high, try again! Guess remaining: 1
    Enter your guess: 80
    Your guess is too low, try again! Guess remaining: 0
    Sorry you are out of guesses. Correct number is 82
    


```python

```
