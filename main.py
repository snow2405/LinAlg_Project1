#Copyright 2023 Lion Six

##############################################################################################################
#   NO EXTERNAL PACKAGES ARE ALLOWED SO CUSTOM IMPLEMENTATION OF FRACTIONS IS REQUIRED
#   TO ENSURE RESULTS ARE MATHEMATICALLY ACCURATE
#   START OF FRACTION IMPLEMENTATION
##############################################################################################################

#Adds two fractions
#frac1: the first fraction as a tuple (numerator, denominator)
#frac2: the second fraction as a tuple (numerator, denominator)
#return: the result of the addition as a tuple (numerator, denominator)
def add_fractions(frac1, frac2):
    return (frac1[0] * frac2[1] + frac2[0] * frac1[1], frac1[1] * frac2[1])

#Subtracts two fractions
#frac1: the first fraction as a tuple (numerator, denominator)
#frac2: the second fraction as a tuple (numerator, denominator)
#return: the result of the subtraction as a tuple (numerator, denominator)
def subtract_fractions(frac1, frac2):
    return (frac1[0] * frac2[1] - frac2[0] * frac1[1], frac1[1] * frac2[1])

#Multiplies two fractions
#frac1: the first fraction as a tuple (numerator, denominator)
#frac2: the second fraction as a tuple (numerator, denominator)
#return: the result of the multiplication as a tuple (numerator, denominator)
def multiply_fractions(frac1, frac2):
    return (frac1[0] * frac2[0], frac1[1] * frac2[1])

#Divides two fractions
#frac1: the first fraction as a tuple (numerator, denominator)
#frac2: the second fraction as a tuple (numerator, denominator)
#return: the result of the division as a tuple (numerator, denominator)
def divide_fractions(frac1, frac2):
    return (frac1[0] * frac2[1], frac1[1] * frac2[0])

#Compute the greatest common divisor of a and b.
# a: the first number (int)
# b: the second number (int)
# return: the greatest common divisor of a and b (int)
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

#Simplifies a fraction to its lowest terms.
#fraction: the fraction to be simplified as a tuple (numerator, denominator)
#return: the simplified fraction as a tuple (numerator, denominator)
def simplify_fraction(fraction):
    numerator, denominator = fraction
    common_divisor = gcd(numerator, denominator)
    return (numerator // common_divisor, denominator // common_divisor)

#Convert float to fraction
#number: the number to be converted (int or float)
#return: the number as a tuple (numerator, denominator)
def convert_to_fraction(number):
    if isinstance(number, int):
        return (number, 1)
    elif isinstance(number, float):
        denominator = 10 ** len(str(number).split('.')[1])
        numerator = int(number * denominator)
        return simplify_fraction((numerator, denominator))
    else:
        raise ValueError("Input must be an integer or a float.")

#Converts all elements of a matrix to fractions.
#matrix: the matrix to be converted (2D list of floats)
#return: the matrix as a 2D list of tuples representing fractions (numerator, denominator)
def convert_matrix_to_fractions(matrix):
    return [[convert_to_fraction(element) for element in row] for row in matrix]

##############################################################################################################
#   End of fraction implementation
##############################################################################################################

##############################################################################################################
#   Start of Function definitions
##############################################################################################################

#this function prints the matrix in a nice format for the user
#matrix: the matrix to be printed (2D list of tuples representing fractions (numerator, denominator)
#return: None (prints the matrix in user friendly form, also makes sure all items are the same width)
def print_matrix(matrix):
    # Determine the length of the longest string representation of each fraction
    max_length = max(len(f"{num}/{denom}" if denom != 1 else f"{num}") for row in matrix for num, denom in row)

    for row in matrix:
        # Format each element to have the same width
        formatted_row = " ".join(f"{num}/{denom}".rjust(max_length) if denom != 1 else f"{num}".rjust(max_length) for num, denom in row)
        print(f"| {formatted_row} |")
    print()

###
### Task 1 (a)
###
#this function is used to find the inverse of a matrix using gaussian elimination
#it uses no additional packages and is implemented from scratch
#matrix: the matrix to be inverted (2D list of tuples representing fractions (numerator, denominator)
#return: the inverted matrix as a 2D list of tuples representing fractions (numerator, denominator)
def gaussian_elimination_for_inverse(matrix):
    num_rows = len(matrix)

    # Create identity matrix
    identity_matrix = [[(1 if i == j else 0, 1) for i in range(num_rows)] for j in range(num_rows)]

    # Convert matrix to reduced row echelon form
    for i in range(num_rows):
        pivot_row = i

        # Find pivot row
        for j in range(i + 1, num_rows):
            if abs(matrix[j][i][0] / matrix[j][i][1]) > abs(matrix[pivot_row][i][0] / matrix[pivot_row][i][1]):
                pivot_row = j

        # Swap pivot row with current row
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
        identity_matrix[i], identity_matrix[pivot_row] = identity_matrix[pivot_row], identity_matrix[i]

        # Reduce current row
        pivot_value = matrix[i][i]
        
        # Divide row by pivot value
        for j in range(num_rows):
            matrix[i][j] = simplify_fraction(divide_fractions(matrix[i][j], pivot_value))
            identity_matrix[i][j] = simplify_fraction(divide_fractions(identity_matrix[i][j], pivot_value))

        # Subtract current row from all other rows
        for j in range(num_rows):
            if j != i:
                factor = matrix[j][i]
                for k in range(num_rows):
                    matrix[j][k] = simplify_fraction(subtract_fractions(matrix[j][k], multiply_fractions(factor, matrix[i][k])))
                    identity_matrix[j][k] = simplify_fraction(subtract_fractions(identity_matrix[j][k], multiply_fractions(factor, identity_matrix[i][k])))

    return identity_matrix

#this function calculates the determinant of a matrix
#matrix: the matrix to calculate the determinant of (2D list of tuples representing fractions (numerator, denominator)
#return: the determinant of the matrix as a tuple (numerator, denominator)
def determinant(matrix):
    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if len(matrix) == 2:
        return subtract_fractions(multiply_fractions(matrix[0][0], matrix[1][1]), multiply_fractions(matrix[0][1], matrix[1][0]))

    # Recursive case for larger matrices
    det = (0, 1)
    for col in range(len(matrix)):
        # Create submatrix for minor
        submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        minor_det = determinant(submatrix)

        # Calculate cofactor
        cofactor = multiply_fractions(matrix[0][col], minor_det)
        if col % 2 == 1:  # Adjust sign for odd columns
            cofactor = multiply_fractions(cofactor, (-1, 1))

        # Add to determinant
        det = add_fractions(det, cofactor)

    return det

###
### Task 1 (b)
###
#this function checks if the matrix is valid for inversion
#matrix: the matrix to be checked (2D list of tuples representing fractions (numerator, denominator)
#return: None (raises an error if the matrix is invalid)
def check_validity_of_matrix(matrix):
    # Check if the matrix is square
    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            raise ValueError("Error: The matrix is not square and therefore not invertible.")

    # Check if the matrix is singular (determinant is zero)
    if determinant(matrix) == (0, 1):
        raise ValueError("Error: The matrix is singular and therefore not invertible.")

    return None

#this function gets the matrix input from the user
#size: the size of the matrix (int)
#return: the matrix as a 2D list of floats
def get_matrix_input(size):
    matrix = []
    print(f"Enter the {size}x{size} matrix, one row at a time with elements separated by spaces:")

    for i in range(size):
        while True:
            row = input(f"Row {i + 1}: ").split()
            if len(row) == size:
                try:
                    matrix_row = [float(element) for element in row]
                    matrix.append(matrix_row)
                    break
                except ValueError:
                    print("Invalid input: Please enter only numbers.")
            else:
                print(f"Invalid input: Each row must have exactly {size} elements.")
    
    return matrix

#this function gets the vector input from the user
#size: the size of the vector (int)
#return: the vector as a list of floats
def get_vector_input(size):
    while True:
        vector = input(f"Enter the right-hand side vector of size {size}, separated by spaces: ").split()
        if len(vector) == size:
            try:
                return [float(element) for element in vector]
            except ValueError:
                print("Invalid input: Please enter only numbers.")
        else:
            print(f"Invalid input: The vector must have exactly {size} elements.")


#This function multiplies a matrix with a vector
#matrix: A 2D list of tuples representing fractions (numerator, denominator)
#vector: A list of tuples representing fractions (numerator, denominator)
#return: The resulting vector as a list of tuples
def matrix_vector_multiply(matrix, vector):
    result_vector = []
    for row in matrix:
        sum = (0, 1)  # Fractional representation of zero
        for elem, vec_elem in zip(row, vector):
            product = multiply_fractions(elem, vec_elem)
            sum = add_fractions(sum, product)
        result_vector.append(simplify_fraction(sum))
    return result_vector


##############################################################################################################
#   End of Function definitions
##############################################################################################################

##############################################################################################################
#   MAIN PROGRAM
##############################################################################################################

###
### Data for Task 1 (c)
###
STATIC_matrices_1c = [
    [[1]],  # 1x1 Invertible
    [[0]],  # 1x1 Singular
    [[1, 2], [3, 4]],  # 2x2 Invertible
    [[1, 2], [2, 4]],  # 2x2 Singular
    [[1, 2, 3], [4, 5, 6], [7, 8, 10]],  # 3x3 Invertible
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],  # 3x3 Singular
    [[1, 0, 2, 3], [0, 1, 4, 5], [6, 7, 0, 1], [2, 3, 8, 9]],  # 4x4 Invertible
    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],  # 4x4 Singular
    [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 5]],  # 5x5 Invertible
    [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]  # 5x5 Singular
]

###
### Task 1 (c)
###
# Test each element in the list of matrices as descirbed in task 1 (c)
for matrix in STATIC_matrices_1c:
    try:
        print(f"TASK 1(c) {STATIC_matrices_1c.index(matrix) + 1} / {len(STATIC_matrices_1c)}:")
        input_matrix = matrix
        input_matrix = convert_matrix_to_fractions(input_matrix)
        print("Given matrix:")
        print_matrix(input_matrix)
        check_validity_of_matrix(input_matrix)  # This will raise an error if the matrix is invalid as described in task 1 (b)

        # Proceed with inverse calculation since the matrix is valid    
        inverse_matrix = gaussian_elimination_for_inverse(input_matrix)
        print("The resulting inverse of this given matrix is:")
        print_matrix(inverse_matrix)
    except ValueError as e:
        print(e)

    print("------------------------------------------------------------------------------------------------------------------")

###
### Task 1(d)
###
try:
    print("\nTASK 1(d):")
    size = int(input("Enter the size of the matrix (e.g., '3' for a 3x3 matrix) or press ENTER to skip to task 2(b): "))
    if size == 0:
        raise ValueError
    matrix = get_matrix_input(size)
    input_matrix = matrix
    input_matrix = convert_matrix_to_fractions(input_matrix)
    print("Given matrix:")
    print_matrix(input_matrix)

    check_validity_of_matrix(input_matrix)  # This will raise an error if the matrix is invalid
    # Proceed with inverse calculation since the matrix is valid
    inverse_matrix = gaussian_elimination_for_inverse(input_matrix)

    vector = get_vector_input(size)
    vector_fraction = convert_matrix_to_fractions([vector])
    print("Given vector:")
    print_matrix(vector_fraction)

    print("General Solution of the equation:")
    result_vector = matrix_vector_multiply(inverse_matrix, vector_fraction[0])
    print_matrix([result_vector])

except ValueError:
    print("Invalid input or no input given.")

print("------------------------------------------------------------------------------------------------------------------")
###
### Task 2(b)
###
STATIC_matirx_2b = [[1, -3, -7],[-1, 5, 6],[-1, 3, 10]]
STATIC_vector_2b = [10, -21, -7]
input_matrix = STATIC_matirx_2b
input_matrix = convert_matrix_to_fractions(input_matrix)
print("\nTASK 2(b):")
print("Given matrix:")
print_matrix(input_matrix)

check_validity_of_matrix(input_matrix)  # This will raise an error if the matrix is invalid
# Proceed with inverse calculation since the matrix is valid
inverse_matrix = gaussian_elimination_for_inverse(input_matrix)

vector = STATIC_vector_2b
vector_fraction = convert_matrix_to_fractions([vector])
print("Given vector:")
print_matrix(vector_fraction)

print("General Solution of the equation:")
result_vector = matrix_vector_multiply(inverse_matrix, vector_fraction[0])
print_matrix([result_vector])

print("------------------------------------------------------------------------------------------------------------------")
###
### Task 3
###
STATIC_matrix_3 = [[1,2,3], [0,1,4], [5,6,0]]
input_matrix = STATIC_matrix_3
input_matrix = convert_matrix_to_fractions(input_matrix)
print("\nTASK 3:")
print("Given matrix:")
print_matrix(input_matrix)
check_validity_of_matrix(input_matrix) # This will raise an error if the matrix is invalid as described in task 1 (b)

# Proceed with inverse calculation since the matrix is valid    
inverse_matrix = gaussian_elimination_for_inverse(input_matrix)
print("The resulting inverse of this given matrix is:")
print_matrix(inverse_matrix)