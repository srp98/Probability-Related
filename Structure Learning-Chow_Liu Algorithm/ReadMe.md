# Structure Learning: Chow-Liu Algorithm Implementation
The program is a implementation of Chow-Liu algorithm in the form of an example to find a structure in cocunut prices in market and determine which to buy.

## Requirements
- Python 3.8+
- Numpy

## Data
`coconut.csv` file should contain 2.8 years of daily coconut oil price movement data quantized to +1(price went up in the past day), 0 (price stayed exactly the same past day) and -1 (price went down in the past day).

Each line should be like this: ```10/9/2012,-1,1,0,1 ``` (month/day/year followed by price movements for markets 0, 1, 2, 3)

## Problem Statement

We are assuming the price movements for markets(0, 1, 2, 3) on different days are modelled to be i.i.d (independent and identically distributed) then we implement chow-liu for learning a tree for these 4 markets.

We then learn the parameters of the tree then inference to get our observations through a sum-product algorithm