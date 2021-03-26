# railcar_packing
Python GUI that takes user-input distributions and finds optimal railcar packing.

Railcar length = 575 feet
Possible Board Lengths (feet): [8, 10, 12, 14, 16, 18 , 20]

GUI takes relative ratios of quantities of each type of board (with associated tolerances) to define a distribution. It then uses this distribution to find quantity configurations that are similar to the distribution (i.e. within the tolerances) while also fitting on the railcar. 

The user defined distribution, best packing configurations, and best match to distribution are all displayed in figures that can be toggled off and on. 
