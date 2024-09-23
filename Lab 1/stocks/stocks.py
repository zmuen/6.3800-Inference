#!/usr/bin/env python
"""
stocks.py

Please read the project instructions beforehand! Your code should go in the
blocks denoted by "YOUR CODE GOES HERE" -- you should not need to modify any
other code!
"""

# import packages here
import numpy as np
import matplotlib.pyplot as plt
import time

# Information for Stocks A and B
priceA = np.loadtxt('data/priceA.csv')
priceB = np.loadtxt('data/priceB.csv')


# DO NOT RENAME OR REDEFINE THIS FUNCTION.
# THE compute_average_value_investments FUNCTION EXPECTS NO ARGUMENTS, AND
# SHOULD OUTPUT 2 FLOAT VALUES
def compute_average_value_investments():
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (b)
    #
    # ASSIGN YOUR FINAL VALUES TO THE RESPECTIVE VARIABLES, I.E.,
    # average_buyandhold = (YOUR ANSWER FOR BUY & HOLD)
    # average_rebalancing = (YOUR ANSWER FOR REBALANCING)
    initial_investment = 1.0
    gamma = 1.05  # Growth factor for Investment C
    beta = 1.4    # Growth factor for Investment D
    alpha = 0.7875 # Shrink factor for Investment D
    days = 20  # Investment window

    # Total number of possible patterns
    num_patterns = 2 ** days

    wealth_buy_and_hold = np.zeros(num_patterns)
    wealth_constant_rebalancing = np.zeros(num_patterns)

    # Buy and Hold Strategy
    def buy_and_hold(binary_pattern, initial_investment):
        worthC = initial_investment / 2  
        worthD = initial_investment / 2  
        
        for day in range(days):
            # Investment C grows by gamma every day
            worthC *= gamma
            
            # Investment D grows or shrinks based on the pattern
            if binary_pattern[day] == '1':
                worthD *= beta
            else:
                worthD *= alpha
        
        # Final wealth after all days
        return worthC + worthD

    # Constant Rebalancing Strategy
    def constant_rebalancing(binary_pattern, initial_investment):
        worthC = initial_investment / 2 
        worthD = initial_investment / 2 
        
        for day in range(days):
            # Investment C grows by gamma every day
            worthC *= gamma
            
            # Investment D grows or shrinks based on the pattern
            if binary_pattern[day] == '1':
                worthD *= beta
            else:
                worthD *= alpha
            
            # Rebalance the total value equally between both investments
            total = worthC + worthD
            worthC = total / 2
            worthD = total / 2
        
        return worthC + worthD

    # Iterate over all possible patterns to compute final wealth for both strategies
    for i in range(num_patterns):
        # Convert integer to binary string representing growth/shrink pattern
        binary_pattern = np.binary_repr(i, width=days)  # Zero-padded binary string of length 'days'
        
        # Compute final portfolio values for both strategies
        wealth_buy_and_hold[i] = buy_and_hold(binary_pattern, initial_investment)
        wealth_constant_rebalancing[i] = constant_rebalancing(binary_pattern, initial_investment)
    
    # Compute the averages across all patterns
    average_buyandhold = np.mean(wealth_buy_and_hold)
    average_rebalancing = np.mean(wealth_constant_rebalancing)
    #
    # END OF YOUR CODE FOR PART (b)
    # -------------------------------------------------------------------------
    return average_buyandhold, average_rebalancing

# DO NOT RENAME OR REDEFINE THIS FUNCTION.
# THE compute_doubling_rate_investments FUNCTION EXPECTS NO ARGUMENTS, AND
# SHOULD OUTPUT 2 FLOAT VALUES
def compute_doubling_rate_investments():
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (d)
    #
    # ASSIGN YOUR FINAL VALUES TO THE RESPECTIVE VARIABLES, I.E.,
    # doubling_rate_buyandhold = (YOUR ANSWER FOR BUY & HOLD)
    # doubling_rate_rebalancing = (YOUR ANSWER FOR REBALANCING)
    import math
    initial_investment = 1.0
    gamma = 1.05
    beta = 1.4 
    alpha = 0.7875 
    days = 20

    # Generate all valid patterns with exactly 10 '1's and 10 '0's
    def generate_patterns(n, ones_count):
        def backtrack(pattern, ones_left, zeros_left):
            if len(pattern) == n:
                patterns.append(pattern)
                return
            
            if ones_left > 0:
                backtrack(pattern + '1', ones_left - 1, zeros_left)
            
            if zeros_left > 0:
                backtrack(pattern + '0', ones_left, zeros_left - 1)
        
        patterns = []
        backtrack("", ones_count, n - ones_count)
        return patterns

    # Generate all patterns with exactly 10 up days and 10 down days
    patterns = generate_patterns(days, days // 2)
    # Buy and Hold Strategy
    def buy_and_hold(pattern, initial_investment):
        worthC = initial_investment / 2
        worthD = initial_investment / 2 
        
        for day in range(days):
            worthC *= gamma
            
            if pattern[day] == '1':
                worthD *= beta
            else:
                worthD *= alpha
        
        return worthC + worthD

    # Constant Rebalancing Strategy
    def constant_rebalancing(pattern, initial_investment):
        worthC = initial_investment / 2  # Start with half in Investment C
        worthD = initial_investment / 2  # Start with half in Investment D
        
        for day in range(days):
            worthC *= gamma
            
            if pattern[day] == '1':
                worthD *= beta
            else:
                worthD *= alpha
            
            # Rebalance the total value equally between both investments
            total = worthC + worthD
            worthC = total / 2
            worthD = total / 2
        
        return worthC + worthD

    # Arrays to store doubling rates for both strategies across all patterns
    doubling_rates_buy_and_hold = np.zeros(len(patterns))
    doubling_rates_constant_rebalancing = np.zeros(len(patterns))

    # Evaluate each pattern
    for i, binary_pattern in enumerate(patterns):
        final_wealth_buy_and_hold = buy_and_hold(binary_pattern, initial_investment)
        final_wealth_constant_rebalancing = constant_rebalancing(binary_pattern, initial_investment)

        # Compute the doubling rate for both strategies
        doubling_rates_buy_and_hold[i] = (1 / days) * math.log2(final_wealth_buy_and_hold)
        doubling_rates_constant_rebalancing[i] = (1 / days) * math.log2(final_wealth_constant_rebalancing)

    # Compute the average doubling rates across all patterns
    doubling_rate_buyandhold = np.mean(doubling_rates_buy_and_hold)
    doubling_rate_rebalancing = np.mean(doubling_rates_constant_rebalancing)
    #
    # END OF YOUR CODE FOR PART (d)
    # -------------------------------------------------------------------------
    return doubling_rate_buyandhold, doubling_rate_rebalancing



def main():
    #
    print("PART (a)")
    #
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (a)
    #
    initial_investment = 1.0
    def buy_and_hold(pricesA, pricesB, initial_investment):
        sharesA = initial_investment / 2 / pricesA[0]
        sharesB = initial_investment / 2 / pricesB[0]
        valueA = sharesA * pricesA
        valueB = sharesB * pricesB
        total_value = valueA + valueB
        return total_value
    
    def constant_rebalancing(pricesA, pricesB, initial_investment):
        total_value = np.zeros(len(pricesA))
        sharesA = initial_investment / 2 / pricesA[0]
        sharesB = initial_investment / 2 / pricesB[0]
        
        for i in range(len(pricesA)):
            current_value = sharesA * pricesA[i] + sharesB * pricesB[i]
            total_value[i] = current_value
            # Rebalance for the next day
            sharesA = (current_value / 2) / pricesA[i]
            sharesB = (current_value / 2) / pricesB[i]
        
        return total_value
    
    AB_wealth_buy_and_hold = buy_and_hold(priceA, priceB, initial_investment)
    AB_wealth_constant_rebalancing = constant_rebalancing(priceA, priceB, initial_investment)

    # Plotting the wealth over time
    plt.figure(figsize=(10, 6))
    plt.plot(AB_wealth_buy_and_hold, label='Buy and Hold', color='blue')
    plt.plot(AB_wealth_constant_rebalancing, label='Constant Rebalancing', color='red')
    plt.title('Investment Strategies Comparison over Time')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    #
    # END OF YOUR CODE FOR PART (a)
    # -------------------------------------------------------------------------

    print("PART (b)")
    average_buyandhold, average_rebalancing = compute_average_value_investments()
    print(f'Computed Averaged Value for Buy & Hold: {average_buyandhold}')
    print(f'Computed Averaged Value for Rebalancing: {average_rebalancing}')
    print()

    #
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (c)
    #

    import math
    def compute_fraction_better_buy_and_hold():
        initial_investment = 1.0
        gamma = 1.05 
        beta = 1.4 
        alpha = 0.7875 
        days = 20

        # Generate all valid patterns with 10 '1's and 10 '0's
        def generate_patterns(n, ones_count):
            def backtrack(pattern, ones_left, zeros_left):
                if len(pattern) == n:
                    patterns.append(pattern)
                    return
                
                if ones_left > 0:
                    backtrack(pattern + '1', ones_left - 1, zeros_left)
                
                if zeros_left > 0:
                    backtrack(pattern + '0', ones_left, zeros_left - 1)
            
            patterns = []
            backtrack("", ones_count, n - ones_count)
            return patterns

        # Generate all patterns with 10 up days and 10 down days
        patterns = generate_patterns(days, days // 2)

        # Buy and Hold Strategy
        def buy_and_hold(pattern, initial_investment):
            worthC = initial_investment / 2
            worthD = initial_investment / 2 
            
            for day in range(days):
                worthC *= gamma
                
                if pattern[day] == '1':
                    worthD *= beta
                else:
                    worthD *= alpha
            
            return worthC + worthD

        # Constant Rebalancing Strategy
        def constant_rebalancing(pattern, initial_investment):
            worthC = initial_investment / 2
            worthD = initial_investment / 2
            
            for day in range(days):
                worthC *= gamma
                
                if pattern[day] == '1':
                    worthD *= beta
                else:
                    worthD *= alpha
                
                # Rebalance the total value equally between both investments
                total = worthC + worthD
                worthC = total / 2
                worthD = total / 2
            
            return worthC + worthD

        # Counters for strategy performance
        buy_and_hold_better_count = 0
        total_patterns = len(patterns)

        # Evaluate each pattern
        for pattern in patterns:
            final_wealth_buy_and_hold = buy_and_hold(pattern, initial_investment)
            final_wealth_constant_rebalancing = constant_rebalancing(pattern, initial_investment)

            rn_buy_and_hold = (1 / days) * math.log2(final_wealth_buy_and_hold)
            rn_constant_rebalancing = (1 / days) * math.log2(final_wealth_constant_rebalancing)

            # Determine which strategy performed better
            if rn_buy_and_hold > rn_constant_rebalancing:
                buy_and_hold_better_count += 1

        # Calculate the fraction of patterns where Buy and Hold performed better
        fraction_better_buy_and_hold = buy_and_hold_better_count / total_patterns

        return fraction_better_buy_and_hold

    fraction_better_buy_and_hold = compute_fraction_better_buy_and_hold()
    print("PART (c)")
    print(f'Fraction of patterns where Buy and Hold performs better: {fraction_better_buy_and_hold}')
    #
    # END OF YOUR CODE FOR PART (c)
    # -------------------------------------------------------------------------

    print("PART (d)")
    doubling_rate_buyandhold, doubling_rate_rebalancing = compute_doubling_rate_investments()
    print(f'Computed Doubling Rate for Buy & Hold: {doubling_rate_buyandhold}')
    print(f'Computed Doubling Rate for Rebalancing: {doubling_rate_rebalancing}')
    print()
    
    print("PART (e)")
    #
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (e)
    #



    
    #
    #
    # END OF YOUR CODE FOR PART (e)
    # -------------------------------------------------------------------------

if __name__ == '__main__':
    main()