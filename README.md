# POS-Tagging

This project was done as a part of CSCI-B-551 Elements of Artificial Intelligence Coursework under Prof. Dr. David Crandall.

## Command to run the program ##

python3 ./label.py training_file testing_file

## Observation

There were 5 functions given in the skeleton code, i.e., posterior (which had if-else statements that calculates posterior probabilities for all 3 parts of the assignment), train, simplified (which was to return the predicted tags depending on simple bayesian network algorithm), hmm_viterbi (to return the predicted tags based viterbi) and complex_mcc (to return result based on mcc calculations). The utility and evaluation functions were given in the initial skeleton code.

## Approach and design decisions

For the simple model, we have initialized the posterior probability as 1 for the ease of multiplication. We have then iterated through each label and word and found the tag and word at the ith index. We have then checked if the word is there in the word emission probability, it its present then we have checked the probability of the tag assigned to that word, and if it is not present or probability of tag of that word is 0 we have taken the probability as 0.000000001. If the posterior comes out to be 0 then we have set the posterior probability to 0.000000001 and then returned the log of the posterior probability.

For the HMM model, just like the Simple model, we have again initialized the posterior probability as 1 for the ease of multiplication and then iterated through each label and word and found the tag and word at the ith index. We have then checked if the word is there in the word emission probability, it its present then we have checked the probability of the tag assigned to that word, and if it is not present or probability of tag of that word is 0 we have taken the probability as 0.000000001. We have then multiplied the probability of tag I with the posterior probability. We have then multiplied each transition probability to compute the posterior probability. If the posterior comes out to be 0 then we have set the posterior probability to 0.000000001 and then returned the log of the posterior probability.

For the Complex model, just like the Simple model, we have again initialized the posterior probability as 1 for the ease of multiplication and then iterated through each label and word. We have set the emission probability as 0.000000001. If the word exists in word emit probability and its value is not 0 then we have set it. We have then set the transition probability from the current state to the next state which was initially set to 0.000000001. Similarly, we have then set the transition probability from the last state to the current state which was initially set to 0.000000001. Now if the index is 0 we have multiplied the transition probability from the current state to the next state with the emission probability and the tag probability of the current tag. If the index is N-1 we have multiplied the transition probability from the last state to the current state with the emission probability and the tag probability of the last tag otherwise we have multiplied the transition probability from the current state to the next state with the emission probability, tag probability of the last tag and the transition probability from the last state to the current state. If the posterior probability comes out to be 0, we have set it to 0.000000001. We have then returned the log of posterior probability.

For the training, we have initialized an empty dictionary for the tag probability. We have used tag as the key and the occurrence as the value. It will be a nested dictionary with key as tag and value as dictionary of tags as key and their transition probability as value. For the word emit probability, it will be a nested dictionary word as key and value as dictionary of tags as key and their probability to occur for a given word as value. Then we have set an initial probability dictionary with tag as key and probability of its occurrence first in sentence as key. We have then set the initial occurrence of a tag, if its already present we have increased it else initialized it. We have then iterated through each tag to set the transition of each tag in the sentence. If the current tag is present in the transition dictionary, we have increased the count by 1 otherwise initialized it to 1, else if the current tag is not present in the transition dictionary, we have initialized its initial dictionary and have set its value to 1. We have then iterated through each tag and increased the count by 1 if it exists otherwise initialized it to 1.

Then we have iterated through each word and set the count of tag for each word. If the word already exists in the word emission dictionary, and if tag exists for the word in the emission dictionary, then we have increased the count by 1 else initialized it to 1, else, we have re initialized the dictionary and have set the occurrence of tag to 1. We have then calculated the tag list and the total number of tags. We have then divided the tag probability with the count of tags and divided the init probability with total length of the data. We have then changed the count of the transition to the transition probability for each tag from ach tag. We have then changed the word emit probability dictionary which is probability of tag if a word is given from its count to probability.

Now moving to the simplified method, if the word is not present in the word emit prob dictionary, we will continue and go to the next iteration. If the word exists, we will initially take the max prob tag as ‘noun’ and set it to 0. No, we have iterated through each word in the word emit prob dictionary for that word, we have found the probability of occurrence of that tag by multiplying emit prob of that tag with the occurrence probability of the tag. If this probability is greater than the max probability, then we have set the tag as maximum probable tag and the maximum probability as the current probability. We have appended the max prob tag to the predicted tag list.

For HMM Viterbi, we have initialized an empty V table to maintain what is the highest probability to reach that state from the previous state. We have also initialized an empty which table to maintain from which previous state we can reach that state with highest probability. We have then found the V table probability for index word as index 0 for each word. Next we have iterated through each tag in the tag list and used the init probability with emit probability to calculate the probability of index 0 in the V table. Next we have found the best tag from the previous state and max probability by which we can reach a particular tag for the word at index i by multiplying probability of each tag from previous state and multiplying it by transition probability and finding the max of it. Initially we have set the tag to ’noun’ and the probability to -1. We then iterate through every key in the V table and finding the value of N-1 index if its greater than probability initially set as -1. WE have returned the predicted tag by finding the predicted tag for index i using the which table with backtracking.

For MCMC, we have written a generator function which we will use for the MCMC method. We have iterated through each tag of the list, if the word exists in the emit prob dictionary we have found its probability from there, we have then calculated transition probabilities from current state to next and previous state to current state. These probabilities are then appended, depending on different location. The sum of all these probabilities is calculated. We have then calculated the cumulative sum. Next, we have used this generator for calling MCMC.

## Challenges

The main challenge was to understand the entire concept of MCMC and apply it in our code. We also had a difficult time while trying to increase the accuracy of our code.

## Results

So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
           
0. Ground truth:       100.00%              100.00%
1. Simple:              92.72%               42.05%
2. HMM:                 94.30%               50.25%
3. Complex:             92.90%               43.15%


