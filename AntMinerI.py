# I have edit the original code of the sectionUses the list of discovered rules to predict the classes of a test set - returns the predictive accuracy. This code is of Ant miner-I which is used in my master thesis.


import sys
import os.path
import math
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
start_time = datetime.now()
# Class for holding all information about a data set
class DataSet:

    def __init__( self ):
        self.data = []
        self.attr_vals = {}
        self.class_attr = ""
        self.attr_idxs = {}
        self.row_table = {}

    def get_attrs( self ):
        return list(self.attr_vals.keys())

# Class for holding rules
class Rule:

    def __init__( self ):
        self.antecedent = {}
        self.consequent = ""
        self.quality = 0

    # Assigns the majority class of the predicted set as the consequent
    def assign_consequent( self, training_set ):
        antecedent_attrs = list(self.antecedent.keys())
        class_freqs = {}.fromkeys(training_set.attr_vals[training_set.class_attr], 0)
        class_attr_idx = training_set.attr_idxs[training_set.class_attr]
        # Get counts of all classes
        for row in training_set.data:
            is_case = True
            for attr in antecedent_attrs:
                if row[training_set.attr_idxs[attr]] != self.antecedent[attr]:
                    is_case = False
                    break
            if is_case:
                class_freqs[row[class_attr_idx]] += 1

        # Get majority class
        majority_count = 0
        self.consequent = random.choice(training_set.attr_vals[training_set.class_attr])
        for clas in list(class_freqs.keys()):
            if class_freqs[clas] > majority_count:
                majority_count = class_freqs[clas]
                self.consequent = clas

    # Computes the quality of this rule based on a training_set
    def compute_quality( self, training_set ):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        class_attr_idx = training_set.attr_idxs[training_set.class_attr]
        antecedent_attrs = list(self.antecedent.keys())
        # Count TPs, TNs, FPs, FNs
        for row in training_set.data:
            is_case = True
            for attr in antecedent_attrs:
                if row[training_set.attr_idxs[attr]] != self.antecedent[attr]:
                    is_case = False
                    break
            if is_case:
                if row[class_attr_idx] == self.consequent:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if row[class_attr_idx] == self.consequent:
                    false_neg += 1
                else:
                    true_neg += 1
        # Compute quality
        if true_pos + false_neg == 0:
            self.quality = 0
        elif true_neg + false_pos == 0:
            self.quality = 1
        else:
            #self.quality = (true_pos / (true_pos + false_pos)) * (true_neg / (true_neg + false_pos))
            self.quality = (true_pos / (true_pos + false_neg)) * (true_neg / (true_neg + false_pos))


            #self.quality = (true_pos / (true_pos + true_neg))**a * (true_pos / (true_pos+false_pos+true_neg+false_neg))**b

    # Iteratively removes attributes from the rule while it improves the quality
    def prune( self, training_set ):
        antecedent_attrs = list(self.antecedent.keys())
        self.compute_quality(training_set)
        original_quality = self.quality
        original_antecedent = self.antecedent.copy()
        if (len(antecedent_attrs)<5):
         for attr in antecedent_attrs:
            self.antecedent.pop(attr, None)
            self.compute_quality(training_set)
            if (self.quality > original_quality):
                self.prune(training_set)
                break
            else:
                self.antecedent = original_antecedent
                self.quality = original_quality

    # Determines if one rule is the same as another
    def equals( self, rule ):
        if (self.consequent == rule.consequent):
            antecedent_attrs_this = list(self.antecedent.keys())
            antecedent_attrs_arg = list(rule.antecedent.keys())
            if (len(set(antecedent_attrs_this) ^ set(antecedent_attrs_this)) == 0):
                for attr in antecedent_attrs_this:
                    if (self.antecedent[attr] != rule.antecedent[attr]):
                        return False
            else:
                return False
        else:
            return False

        return True

    # Prints a well formatted representation of the rule
    def print_rule( self, class_attr ):
        #print("{ "),
        print("{ ", end = "")
        antecedent_attrs = list(self.antecedent.keys())
        num_attrs = len(antecedent_attrs)
        for t in range(0, num_attrs):
            print(antecedent_attrs[t] + " = " + self.antecedent[antecedent_attrs[t]], end = "")
            #print(antecedent_attrs[t] + " = " + self.antecedent[antecedent_attrs[t]]),
            if t < num_attrs - 1:
                print(" AND ", end = "")
                #print(" AND "),
        print(" } => " + class_attr + " = " + self.consequent)

# Gets the base pheromone level
def gen_base_pheromone_level( attr_vals ):
    total_values = 0
    attrs = list(attr_vals.keys())
    # Count the number of values
    for attr in attrs:
        total_values += len(attr_vals[attr])
    return 1 / total_values

# Returns a pheromone table initialized with the base level of pheromone
def gen_base_pheromone_table( attr_vals ):
    base_pheromone_level = gen_base_pheromone_level(attr_vals)
    attrs = list(attr_vals.keys())
    # Construct dict in the form { attribute : { value : pheromone level } }
    base_pheromone_table = {}.fromkeys(attrs, {})
    for attr in attrs:
        new_sub_table = {}.fromkeys(attr_vals[attr], base_pheromone_level)
        base_pheromone_table[attr] = new_sub_table

    return base_pheromone_table

# Computes the base heuristic values for each value based on the current state of the training set
def gen_heuristic_table( training_set ):
    attrs = training_set.get_attrs()


    # Construct dict in the form { attribute : { value : heuristic } }
    heuristic_table = {}.fromkeys(attrs, {})
    heuristic_table.pop(training_set.class_attr, None)
    class_attr_idx = training_set.attr_idxs[training_set.class_attr]
    classes = training_set.attr_vals[training_set.class_attr]
    # Iterate through attributes
    for a in range(0, len(attrs)):

        curr_attr = attrs[a]
        # Ensure only non-class attributes are computed for heuristics
        if curr_attr != training_set.class_attr:
            curr_vals = training_set.attr_vals[curr_attr]
            # Construct a sub table for an attribute of from { value : heuristic }
            new_heuristic_sub_table = {}.fromkeys(curr_vals, 0)
            new_row_sub_table = {}.fromkeys(curr_vals, [])
            # Iterate through each value of the current values
            for val in curr_vals:
                # Table to hold the frequency of each class
                class_freqs = {}.fromkeys(classes, 0)
                # Count of each case containing the current value
                val_freq = 0
                # Track the rows in which this value occurs
                row_indices = []
                # Iterate through each tuple (row) of the data set and count frequencies
                for r in range(0, len(training_set.data)):

                    curr_row = training_set.data[r]
                    if curr_row[a] == val:
                        val_freq += 1
                        row_indices += [r]
                        class_freqs[curr_row[class_attr_idx]] += 1
                # Compute the heuristic value
                if val_freq > 0:
                    heuristic = 0
                    for clas in classes:
                        freq = class_freqs[clas]
                        # Zero frequency indicates 0 information
                        if freq != 0:
                            #test_vals=  freq/val_freq
                            heuristic -= freq * math.log2(freq)
                    heuristic /= val_freq
                    #heurisctic1=  math.log2(freq) - heuristic

                    # Add to sub table
                    new_heuristic_sub_table[val] = heuristic
                    new_row_sub_table[val] = row_indices
                # If a value no longer occurs in the set, remove it from the heuristic function and the training set
                else:
                    new_heuristic_sub_table.pop(val, None)
                    training_set.attr_vals[curr_attr].remove(val)
                    new_row_sub_table.pop(val, None)
            # Add sub table to heuristic table
            heuristic_table[curr_attr] = new_heuristic_sub_table
            training_set.row_table[curr_attr] = new_row_sub_table

    return heuristic_table

# Gets the number of cases that satisfy a given antecedent. Also updates the heuristic table.
def get_num_cases( training_set, heuristic_table, antecedent, new_attr, new_term ):
    # Make a "test antecedent" with the new term appended
    test_antecedent = antecedent.copy()
    test_antecedent[new_attr] = new_term
    rule_attrs = test_antecedent.keys()

    class_attr_idx = training_set.attr_idxs[training_set.class_attr]
    classes = training_set.attr_vals[training_set.class_attr]
    class_freqs = {}.fromkeys(classes, 0)

    # Get the common indices to avoid searching the entire data set
    row_indices = []
    for attr in rule_attrs:
        row_indices = set(row_indices) ^ set(training_set.row_table[attr][test_antecedent[attr]])

    case_count = 0
    # Check if antecedent covers each row
    for row_idx in row_indices:
        is_case = True
        for attr in rule_attrs:
            if training_set.data[row_idx][training_set.attr_idxs[attr]] != test_antecedent[attr]:
                is_case = False
                break
        if is_case:
            case_count += 1
            class_freqs[training_set.data[row_idx][class_attr_idx]] += 1

    # Update the heuristic value
    if case_count > 0:
        heuristic = 0
        for clas in classes:
            freq = class_freqs[clas]
            # Zero frequency indicates 0 information
            if freq != 0:
                heuristic -= freq * math.log2(freq)
        heuristic /= case_count
        heuristic_table[new_attr][new_term] = heuristic

    return case_count

# Checks the number of cases of an antecedent appended with a new attr against the min_cases_per_rule
def update_usable_attrs (training_set, heuristic_table, antecedent, min_cases_per_rule, usable_attrs ):
    for attr in list(usable_attrs.keys()):
        for term in usable_attrs[attr]:
            if get_num_cases(training_set, heuristic_table, antecedent, attr, term) < min_cases_per_rule:
                usable_attrs[attr].remove(term)
        # If no values for an attribute are usable, remove the attribute entirely
        if len(usable_attrs[attr]) == 0:
            usable_attrs.pop(attr, None)
    return usable_attrs

# Constructs a rule for an ant based on the current state of the pheromone table
def gen_rule( training_set, pheromone_table, heuristic_table,
              min_cases_per_rule, exploitation_tradeoff, heuristic_tradeoff):
    rule = Rule()
    # A table of usable attributes and usable values within those attributes
    usable_attrs = {}.fromkeys(training_set.attr_vals.keys())
    usable_attrs.pop(training_set.class_attr, None)
    for attr in usable_attrs:
        new_list = []
        usable_attrs[attr] = training_set.attr_vals[attr][:]
    unused_attrs = training_set.get_attrs()[:]
    unused_attrs.remove(training_set.class_attr)
    listempty = False
    while len(usable_attrs) > 0:
        while True:
            # Pick a random attribute
            new_term_attr = random.choice(list(usable_attrs.keys()))
            # Pick random term
            if len(usable_attrs[new_term_attr]) > 0:
                new_term = random.choice(usable_attrs[new_term_attr])
            else:
                listempty = True
                break

            # Attempt exploration
            if (random.random() <= exploitation_tradeoff):
                rule.antecedent[new_term_attr] = new_term
                usable_attrs.pop(new_term_attr, None)
                unused_attrs.remove(new_term_attr)
                break
            # Attempt exploitation
            else:
                prob_numerator = (heuristic_tradeoff * pheromone_table[new_term_attr][new_term]) - ((1 - heuristic_tradeoff) * heuristic_table[new_term_attr][new_term])
                prob_denominator = 0
                for attr in unused_attrs:
                    for term in training_set.attr_vals[attr]:
                        prob_denominator += (heuristic_tradeoff * pheromone_table[attr][term]) - ((1 - heuristic_tradeoff) * heuristic_table[attr][term])
                if prob_denominator:
                    prob = prob_numerator / prob_denominator
                else:
                    prob = 1
                # Attempt to add term
                if (random.random() <= prob):
                    rule.antecedent[new_term_attr] = new_term
                    usable_attrs.pop(new_term_attr, None)
                    unused_attrs.remove(new_term_attr)
                    break
        if listempty:
            break
        # Update usable attrs (also updates the heuristic table)
        usable_attrs = update_usable_attrs(training_set, heuristic_table, rule.antecedent, min_cases_per_rule, usable_attrs)

    rule.assign_consequent(training_set)
    return rule


# Updates the pheromone levels of terms covered by the antecedent of a rule
def update_pheromone_table( pheromone_table, rule, evaporation_rate, adjusting_param):
    antecedent_attrs = list(rule.antecedent.keys())
    for attr in antecedent_attrs:
        term = rule.antecedent[attr]
        pheromone_table[attr][term] = (1 - evaporation_rate) * pheromone_table[attr][term] + \
                                      (adjusting_param * math.exp(-evaporation_rate) + rule.quality) \
                                      * pheromone_table[attr][term]

# Removes all cases in the training set covered by a given rule
def remove_covered_cases( training_set, rule ):
    class_attr_idx = training_set.attr_idxs[training_set.class_attr]
    antecedent_attrs = list(rule.antecedent.keys())
    new_data = training_set.data[:]
    for row in training_set.data:
        is_case = True
        for attr in antecedent_attrs:
            if row[training_set.attr_idxs[attr]] != rule.antecedent[attr]:
                is_case = False
                break
        if is_case and row[class_attr_idx] == rule.consequent:
            new_data.remove(row)
    training_set.data = new_data

# Generates a default rule for when a test cases is not covered by any discovered rules - simply takes the majority
# class of the uncovered cases
def gen_default_rule( training_set ):
    class_attr_idx = training_set.attr_idxs[training_set.class_attr]
    classes = training_set.attr_vals[training_set.class_attr]
    class_freqs = {}.fromkeys(classes, 0)
    # Count the occurences of each class
    for row in training_set.data:
        class_freqs[row[class_attr_idx]] += 1
    max_freq = 0
    majority_class = list(class_freqs.keys())[0]
    # Find most frequent class
    for clas in class_freqs:
        if class_freqs[clas] > max_freq:
            max_freq = class_freqs[clas]
            majority_class = clas

    # Generate a rule with an empty antecedent and the majority class as teh consequent
    default_rule = Rule()
    default_rule.consequent = majority_class

    return default_rule

# Performs the Ant Miner (I) algorithm - returns a list of discovered rules
def ant_mine( training_set, num_ants, min_cases_per_rule, max_uncovered_cases, num_rules_converge,
              exploitation_tradeoff, heuristic_tradeoff, evaporation_rate, adjusting_param ):
    uncovered_cases = len(training_set.data)
    base_pheromone_table = gen_base_pheromone_table(training_set.attr_vals)
    discovered_rules = []
    # Generate rules until the max number of uncovered cases remains
    while uncovered_cases > max_uncovered_cases:
        ant_count = 0
        converge_count = 0
        pheromone_table = base_pheromone_table
        rules = []
        prev_rule = Rule()
        # Loop breaks when all ants are used or when the ants converge to a rule
        while True:
            ant_count += 1
            # Reset heuristic table
            heuristic_table = gen_heuristic_table(training_set)
            # Generate a rule
            new_rule = gen_rule(training_set, pheromone_table, heuristic_table, min_cases_per_rule,
                     exploitation_tradeoff, heuristic_tradeoff)
            # Prune the generated rule
            new_rule.prune(training_set)
            # Update the pheromone table based on the rule
            update_pheromone_table(pheromone_table, new_rule, evaporation_rate, adjusting_param)
            # Check for convergence
            if new_rule.equals(prev_rule):
                converge_count += 1
            else:
                converge_count = 0
                rules += [new_rule]
            if ant_count >= num_ants or converge_count >= num_rules_converge: break
        # Get the best rule from this iteration
        best_rule = Rule()
        for rule in rules:
            if rule.quality > best_rule.quality:
                best_rule = rule
        # Add to list of discovered rules
        discovered_rules += [best_rule]
        # Remove cases in the data set covered by the rule
        remove_covered_cases(training_set, best_rule)
        uncovered_cases = len(training_set.data)

    # Add the default rule to the discovered rules
    discovered_rules += [gen_default_rule(training_set)]
    print("Length of Discovered rules",len(discovered_rules))
    return discovered_rules

# Uses the list of discovered rules to predict the classes of a test set - returns the predictive accuracy
def predict_test_data( test_set, discovered_rules ):
    #TP = 0
    # FP = 0
    # TN = 0
    # FN = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    #correct_predictions = 0
    class_attr = test_set.class_attr
    #print('1',class_attr)
    classes = test_set.attr_vals[class_attr]
    # discovered_class_attr=discovered_rules.class_attr
    # discovered_classes = discovered_rules.attr_vals[discovered_class_attr]
    # print("discover rule classes",discovered_classes)
    #print('2',classes)

    confusion_matrix = {}.fromkeys(classes)
    #print("3 confusion Matrix", cclassesonfusion_matrix)
    for clas in classes:
        confusion_matrix[clas] = {}.fromkeys(classes, 0)
        #print(confusion_matrix[clas])
        #print('4',confusion_matrix[clas])

    for row in test_set.data:
        # Iterate through rules until one of the discovered rules fits the case
        for rule in discovered_rules:
            rule_attrs = rule.antecedent.keys()
            is_case = True
            for attr in rule_attrs:
                if row[test_set.attr_idxs[attr]] != rule.antecedent[attr]:
                    is_case = False
                    break

            if is_case:
                if rule.consequent == row[test_set.attr_idxs[class_attr]]:
                    true_pos += 1
                   
                else:
                    false_pos += 1

                    
            else:
                if rule.consequent == row[test_set.attr_idxs[class_attr]]:
                    false_neg += 1
              
                else:
                    true_neg += 1
                    
            

                break

    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    precision=true_pos/(true_pos+false_pos)
    recall=true_pos/(true_pos+false_neg)
    F1_score=2*((precision*recall)/(precision+recall))
    New_F1_score= 2*(true_pos)/(2*(true_pos)+false_neg+false_pos)
    print("Accuracy = %f" % accuracy)
    print("F1-Score = %f" % F1_score)
    print("New F1-Score = %f" % New_F1_score)
    print("Precision = %f" % precision)
    print("Recall = %f" % recall)






def read_data( set_fn, remove_incomplete ):
    set_file = open(set_fn)
    set = DataSet()
    attrs = set_file.readline().replace(',', ' ').split()
    num_attrs = len(attrs)
    set.attr_vals.fromkeys(attrs)
    # Create a table of attribute indices for easy access
    for a in range(0, num_attrs):
        set.attr_idxs[attrs[a]] = a

    while True:
        line = set_file.readline()
        if line == "": break
        vals = line.replace(',', ' ').split()
        #print(len(vals))
        #print(num_attrs)
        if len(vals) == num_attrs and not (remove_incomplete and "?" in vals):
            set.data.append(vals)
            for i in range(0, len(attrs)):
                currVal = vals[i]
                currAttr = attrs[i]
                currValList = set.attr_vals.get(currAttr)
                if currValList is None:
                    currValList = [ currVal ]
                    set.attr_vals[currAttr] = currValList
                elif currVal not in currValList:
                    currValList += [ currVal ]
                    set.attr_vals[currAttr] = currValList

    return set

# General function to be called when an error is encountered
def quit( message ):
    print("Error: %s" % message)
    sys.exit(0)

# Main function: reads all necessary input, validates the input, and calls the algorithm
def main():
    # Read training set filename
    training_set_fn = input("Please enter the name of the training set file: ")
    # Ensure the file exists
    if not os.path.isfile(training_set_fn):
        quit("file does not exist")

    # Read data into a data_set object
    training_set = read_data(training_set_fn, True)

    # Get the class attribute from the user
    attrs = list(training_set.attr_vals.keys())
    num_attrs = len(attrs)
    print("Attributes:")
    for a in range(0, num_attrs):
        print("%d. %s" % (a, attrs[a]))
    class_attr_idx = int(input("Please enter the number corresponding to the class attribute [0-%d]: " \
                               % (num_attrs - 1)))
    if class_attr_idx not in range(0, num_attrs):
        quit("value entered for class attribute is invalid")
    training_set.class_attr = attrs[class_attr_idx]

    # Read in all remaining parameters and ensure they are valid
    num_ants = int(input("Please enter the number of ants (1000 recommended): "))
    if num_ants <= 0:
        quit("value entered for number of ants is non-positive")

    min_cases_per_rule = int(input("Please enter the minimum number of cases per rule (10 recommended): "))
    if min_cases_per_rule <= 0:
        quit("value entered for minimum number of cases per rule is non-positive")

    max_uncovered_cases = int(input("Please enter the maximum number of uncovered cases in the training set (10 recommended): "))
    if max_uncovered_cases < 0:
        quit("value entered for maximum number of uncovered cases is non-zero")

    num_rules_converge = int(input("Please enter the number of rules used to test convergence of the ants (40 recommended): ")) #10
    if num_rules_converge <= 0:
        quit("value entered for number of rules used to test convergence of the ants is non-positive")

    exploitation_tradeoff = 0.02#float(input("Please enter the value for exploitation/exploration tradeoff: ")) #0.01
    #if exploitation_tradeoff < 0 or exploitation_tradeoff > 1:
     #   quit("value entered for exploitation/exploration tradeoff is not normalized")

    heuristic_tradeoff = 0.7#float(input("Please enter the value for trail/visibility tradeoff: ")) #0.8

    evaporation_rate = 0.20#float(input("Please enter the evaporation rate [0-1]: ")) #0.10
    #if evaporation_rate < 0 or evaporation_rate > 1:
     #   quit("value entered for evaporation rate is not normalized")

    adjusting_param =0.85#float(input("Please enter the adjusting parameter [0.8-1]: ")) #0.85
    #if adjusting_param < 0.8 or adjusting_param > 1:
     #   quit("value entered for evaporation rate is not in the indicated range")

    # Call the ant miner algorithm
    discovered_rules = ant_mine(training_set, num_ants, min_cases_per_rule, max_uncovered_cases, num_rules_converge,
              exploitation_tradeoff, heuristic_tradeoff, evaporation_rate, adjusting_param)

    # print all discovered rules
    for rule in discovered_rules:
        rule.print_rule(training_set.class_attr)

    # Read training set filename
    test_set_fn = input("Please enter the name of the test set file: ")
    # Ensure the file exists
    if not os.path.isfile(test_set_fn):
        quit("file does not exist")

    # Read data into a data_set object
    test_set = read_data(test_set_fn, False)
    test_set.class_attr = training_set.class_attr

    predict_test_data(test_set, discovered_rules)
if __name__ == "__main__":
    main()
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
