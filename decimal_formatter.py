# This program is used to print decimal numbers without the e for scientific notation

import decimal

# This function takes any float and prints it without scientific notation
def print_without_e(some_float):
    
    print(some_float)
    
    # Get the string version of our decimal
    str_decimal = decimal.Decimal(str(some_float))
    
    # Get the number of decimal digits to the right of the decimal point
    decimal_count = abs(str_decimal.as_tuple().exponent)
    
    # Get the format string for printing
    format_str = '{0:.' + str(decimal_count) + 'f}'

    # Print some_float formatted to the custom number of digits to the
    # right of the decimal point so it prints correctly
    print(format_str.format(some_float))