import numpy as np
import matplotlib.pyplot as plt
import sympy
import re
import itertools
import math

# For a group presentation < x_1, ..., x_n | r_1, ..., r_m > we define a class object that has two attributes:
# - "generatorCount":   the number of generators for the given presentation, in this case "n".
# - "relations":        a list consisting of the relations r_1, ...,r_m where each word r_i is given as a list of elements [k,l] representing the monomials of r_i.
#                       The k stands for the k-th generator and l is its power.
#
# Example: Given the presentation < x_1, x_2 | x_2^2x_1^5x_2^{-3}, x_1 > we would have 
# - generatorCount = 2
# - relations = [[[2,2],[1,5],[2,-3]],[[1,1]]]
#
# We have the following list of methods for this class:
# - str: returns the group presentation as a string in redable way, i.e. "< x_1, ..., x_n | r_1, ..., r_m >"
# - abelianmatrix: returns a matrix where the columns are given by the images of the relations under the canonical map < x_1, ..., x_n > -> \Z^n sending x_i to the standard basis vectors e_i
# - rank: returns the rank of the group (defined as the dimension of the tensor product of its abelianization with the rational numbers)
# - canonabhom: returns some epimorphism onto a free abelian group of the same rank as a list of the images of the generators under this epimorphism
# - jacobimatrix: returns the Jacobi matrix corresponding to the presentation, i.e. the entry (i,j) of this matrix is given by the Fox derivative of the i-th relation with respect to the j-th generator
class group:

    def __init__(self , g, r):
        self.generatorCount = g
        self.relations = r

    def __str__(self):
        outputstr = "< "
        for i in range(1,self.generatorCount):
            outputstr += f"x_{i}, "
        outputstr += f"x_{self.generatorCount} | "
        for i,word in enumerate(self.relations):
            for letter in word:
                outputstr += f"x_{letter[0]}^{letter[1]} "
            if i < len(self.relations) -1 : outputstr += ", " 
        outputstr += ">"
        return outputstr
    
    def abelianmatrix(self):
        matrix = [[0 for _ in self.relations] for _ in range(0,self.generatorCount)]
        for i,word in enumerate(self.relations):
            for letter in word:
                matrix[letter[0]-1][i] += letter[1]
        return matrix
    
    def rank(self):
        # We can find the rank of a group by counting the the number of zeros on the diagonal of the Smith normal form of the matrix returned by the method "abelianmatrix".
        S,A,T = smithnormalform(self.abelianmatrix())
        n= np.count_nonzero(np.diagonal(A))
        return self.generatorCount - n

    def canonabhom(self):
        # The matrix returned by "abelianmatrix" describes the homomorphism onto its abelianization, hence we get an epimorphism to a free abelian group of the same rank by a suitable projection of the first base change matrix coming from the Smith normal form.
        S,A,T = smithnormalform(self.abelianmatrix())
        n= np.count_nonzero(np.diagonal(A))
        M = np.zeros((self.generatorCount - n, self.generatorCount),np.int64)
        for i in range(0,self.generatorCount - n):
            M[i,n+i] = 1
        S = sympy.Matrix(S)
        return np.linalg.matrix_transpose(sympy.Matrix(M) @ S.inv())

    def jacobimatrix(self):
        x = list(sympy.symbols('x0:%d'%self.generatorCount))
        J = [[foxderivative(word,i,x) for i,symbol in enumerate(x)] for word in self.relations]
        return J

# Given a word in a group and an index for a generator this function computes the fox derivative of the word with respect to this generator. We only need this for the method "jacobimatrix" in the class group.
# Uses sympy.
def foxderivative(word,generatorindex,listofsymbols):
    if len(word) == 0:
        return 0
    elif len(word) == 1:
        # The Fox derivative is uniquely determined by the fact that it is a homomorphism and the following rules:
        if word[0][0]-1 == generatorindex:
            if word[0][1] == 0:
                return 0
            elif word[0][1] > 0:
                return sum([listofsymbols[generatorindex]**k for k in range(0, word[0][1])])
            elif word[0][1] < 0:
                return -sum([listofsymbols[generatorindex]**k for k in range(word[0][1],0)])
        else:
            return 0
    else:
        return foxderivative([word[0]],generatorindex,listofsymbols) + listofsymbols[word[0][0]-1]**word[0][1] * foxderivative(word[1:],generatorindex,listofsymbols)

# Given a homomorphism from a group to a free abelian group this function returns the image of the Jacobi matrix under the induced homomorphism on group rings.
# Uses sympy.
def inducedjacobimatrix(group, hom):
    x = list(sympy.symbols('x0:%d'%group.generatorCount))
    t = list(sympy.symbols('t0:%d'%len(hom[0])))
    matrix = sympy.Matrix(group.jacobimatrix())
    for i in range(0,group.generatorCount):
        image = math.prod([t[line]**entry for line,entry in enumerate(hom[i])])
        matrix = matrix.subs(x[i], image)
    return matrix

# Returns the numerator and denominator of the Alexander function corresponding to a non-trivial group homomorphism. In the case that the homomorphism is trivial the returned value is "None". It is well-defined up to multiplication with monomials.
# Uses sympy and numpy
def alexanderfunction(group, hom):
    t = list(sympy.symbols('t0:%d'%len(hom[0])))
    matrix = inducedjacobimatrix(group, hom)
    arr = []
    k = group.generatorCount
    l = len(group.relations)
    for i in range(0, k):
        if sympy.Matrix(hom[i]).is_zero_matrix == False: 
            if l > 0:
                matrix = np.delete(matrix, i,1) # delete ith column
            for comb in itertools.combinations(list(range(0,l)), k-1):
                arr.append(sympy.det(sympy.Matrix(matrix[comb,:])))
            numerator = sympy.simplify(sympy.gcd(arr))
            denominator = sympy.prod([t[line]**entry for line,entry in enumerate(hom[i])]) - 1
            return numerator,denominator
    
    return None
           
# Plots the coeffiecents of the formal power series corresponding to the Alexander function up to degree N
# Uses sympy and matplotlib
def plotalexanderfunction(group,hom,N):
    t = list(sympy.symbols('t0:%d'%len(hom[0])))
    z = sympy.symbols("z")
    numerator,denominator = alexanderfunction(group, hom)
    # We first turn the multivariable Alexander function into a rational function with one variable.
    for i in range(0, len(hom[0])):
        numerator = numerator.subs(t[i],z)
        denominator = denominator.subs(t[i],z)
    degree = max(sympy.Poly(numerator, z, 1/z).degree_list()+sympy.Poly(denominator, z, 1/z).degree_list())
    numerator = numerator * z**(2*degree)
    denominator = denominator * z**degree
    if group.rank() == 1:
        polynomial = sympy.simplify((z-1) * numerator/denominator) #This is just a stylistic choice which is common for fundamental groups of knot complements.
    else:
        polynomial = sympy.simplify(numerator/denominator)

    coefficients = [sympy.simplify(polynomial.diff(z,k).subs(z,0)/sympy.factorial(k)) for k in range(0,N+1)]
    X = range(0, len(coefficients))

    plt.bar(X,coefficients,label = f"Alexander function corresponding to {hom}")
    plt.legend()
    plt.show()

# This function was used for testing purposes. It returns true if a map from a group to a real vector space (or a subgroup) given by an array of the images of the generators is a well-defined group homomorphism.
# Uses numpy.
def checkgrphom(g,arr):
    a = [sum([char[1] * arr[char[0]-1] for char in word]) for word in g.relations]
    return np.isclose(a,np.zeros(len(g.relations))).all()

# This function returns the row and column of some minimal non-zero entry of a matrix with respect to the absolute value. If no such entry exists it will return "None".
# Uses numpy.
def argabsminnonzero(matrix):
    matrix = np.array(matrix)
    if (matrix == np.zeros(matrix.shape)).all():
        return None
    else:
        matrix = matrix.astype('float')
        # We want to use the numpy function "nanargmin" for which we set all entries with value zero to "np.nan".
        matrix[matrix == 0] = np.nan
        return np.unravel_index(np.nanargmin(np.absolute(matrix)), matrix.shape)

# Given a integer matrix M this function returns three matrices S,A,T such that
# - A is a diagonal matrix with unique entries of increasing order (i.e. the Smith normal form of M)
# - we have M = S * A * T (i.e. S and T are the base change matrices)
# Uses numpy.
def smithnormalform(matrix):
    A = np.array(matrix,np.int64)
    rowCount, columnCount = A.shape
    S = np.identity(rowCount,np.int64)
    T = np.identity(columnCount,np.int64)

    if (A == np.zeros((rowCount,columnCount))).all():
        return S,A,T
    else:
        while not (A[1:rowCount,0:1] == np.zeros((rowCount-1,1))).all() or not (A[0:1,1:columnCount] == np.zeros((1,columnCount-1))).all():
            # First switch the smallest entry of the matrix to the position (0,0)
            index = argabsminnonzero(A)
            temp = np.copy(A[0,:])
            A[0,:] = A[index[0],:]
            A[index[0],:] = temp
            temp = np.copy(A[:,0])
            A[:,0] = A[:,index[1]]
            A[:,index[1]] = temp
            temp = np.copy(S[:,0])
            S[:,0] = S[:,index[0]]
            S[:,index[0]] = temp
            temp = np.copy(T[0,:])
            T[0,:] = T[index[1],:]
            T[index[1],:] = temp

            # Now we try to bring all entries in the first row and column apart from (0,0) to zero
            for r in range(1,rowCount):
                p=A[r,0] // A[0,0]
                A[r,:] = A[r,:] + -p * A[0,:]
                S[:,0] = S[:,0] + p * S[:,r]
            for c in range(1,columnCount):
                p=A[0,c] // A[0,0]
                A[:,c] = A[:,c] + -p * A[:,0]
                T[0,:] = T[0,:] + p * T[c,:]

        # Then we continue with the smaller matrix
        B = A[1:rowCount, 1:columnCount]
        s = np.identity(rowCount,np.int64)
        t = np.identity(columnCount,np.int64)
        a = np.zeros((rowCount,columnCount),np.int64)
        s[1:rowCount, 1:rowCount], a[1:rowCount,1:columnCount], t[1:columnCount, 1:columnCount] = smithnormalform(B)

        # Finally we bring the acquired data together
        a[0,0] = A[0,0]
        S = S @ s
        T = t @ T

        return S,a,T

# This function returns a group from user input.
def getinput():
    usePredefinedGroups = input("Would you like to use a predefined group? (y/n/help) \n")
    selectString = "Here is a list of available groups. Select a group by typing in the corresponding number. \n"
    selectString += "1) Torus knot groups \n"
    selectString += "2) Free groups \n"
    selectString += "3) Free abelian groups \n"
    selectString += "4) Orientable surface groups \n"
    selectString += "5) Non-orientable surface groups \n"

    if usePredefinedGroups == "y":
        groupIndex = input(selectString)
        match groupIndex:
            case "1":
                p,q = input("Please input two coprime integers \n").split()
                if np.gcd(int(p),int(q)) == 1:
                    g = group(2, [[[1,int(p)],[2,-int(q)]]])
                else: 
                    raise RuntimeError("These integers are not coprime. Please restart the program.")
            case "2":
                n = input("Please input a natural number \n")
                if int(n) > 0:
                    g = group(int(n),[])
                else: 
                    raise RuntimeError("This is not a natural number. Please restart the program.")
            case "3":
                n = input("Please input a natural number \n")
                if int(n) > 0:
                    rel = []
                    for i in range(1,int(n)):
                        rel.append([[i, 1],[i+1,1],[i,-1],[i+1,-1]])
                    g = group(int(n),rel)
                else: 
                    raise RuntimeError("This is not a natural number. Please restart the program.")
            case "4":
                n = input("Please input a natural number \n")
                if int(n) > 0:
                    word = []
                    for i in range(1,int(n)+1):
                        word += [[2*i-1, 1],[2*i,1],[2*i-1,-1],[2*i,-1]]
                    g = group(2*int(n),[word])
                else: 
                    raise RuntimeError("This is not a natural number. Please restart the program.")
            case "5":
                n = input("Please input a natural number \n")
                if int(n) > 0:
                    word = []
                    for i in range(1,int(n)+1):
                        word += [[i, 2]]
                    g = group(int(n),[word])
                else: 
                    raise RuntimeError("This is not a natural number. Please restart the program.")
            case _:
                raise RuntimeError("This does not correspond to a group. Please restart the program.")
            
    elif usePredefinedGroups == "n": 
        # The path to the input file depends on the location of the virtual enviroment!
        try:
            with open("inputgroup.txt", "r") as handle: 
                n = handle.readline()
                if int(n) <= 0: 
                    raise RuntimeError("There seems to be a problem with your input file. Follow the instructions and restart the program.")
                relators = []
                for line in handle:
                    rx = re.compile(r'-?\d+')
                    l = rx.findall(line)
                    if len(l) % 2 != 0:
                        raise RuntimeError("There seems to be a problem with your input file. Follow the instructions and restart the program.")
                    word = []
                    for i in range(0,len(l)):
                        if i % 2 == 0: 
                            if int(l[i]) <= 0 or int(l[i]) > int(n):
                                raise RuntimeError("There seems to be a problem with your input file. Follow the instructions and restart the program.")
                            word.append([int(l[i]),int(l[i+1])])
                    relators.append(word)
                g = group(int(n), relators)
        except: 
            raise RuntimeError("There seems to be a problem with your input file. Follow the instructions and restart the program.")
        
    elif usePredefinedGroups == "help":
        print("When starting this program you can choose between using a predefined group or using your own via a text file.")
        print("If you want to use a group of your choice you have to create a file named \"inputgroup.txt\" which stores the information of the presentation.")
        with open("inputgroup_example.txt","w") as handle:
            outputstring = "6 #This is the number of generators for the group. It must be a natural number. The next lines are the relations.\n"
            outputstring += "6,-1 2,1 6,1 1,-1 #They are pairs of numbers separeted by space. The first number of each pair is the index of the generator and the second number is its power.\n"
            outputstring += "1,1 3,1 1,-1 4,-1 #This line corresponds to the word x_1^1*x_3^1*x_1^-1*x_4^-1.\n"
            outputstring += "3,1 5,1 3,-1 6,-1\n2,-1 4,1 2,1 3,-1\n"
            outputstring += "5,1 1,1 5,-1 2,-1 #Make sure that this is the last line."
            handle.write(outputstring)
        print("You can open the file \"inputgroup_example.txt\" to see how this works or read the comment before the class group.")
        print("In the case that you want to use a predefined group you get to the following menu: \n")
        print(selectString)
        print("Now you can choose from the class of groups listed above. Let's say you want to pick a free group, so you would type in: \n2 \n")
        print("The program now asks for the rank of the free group: \n")
        print("Please input a natural number \n")
        print("For example you could write: \n1 \n")
        print("It will now list different properties of this group (presentation) including the Alexander function: \n")
        g = group(1,[])

    else: 
        raise RuntimeError("This is not a valid choice. Please restart the program and use either \"y\", \"n\" or \"help\"!")
    
    return g

# This is the main function that gets called when starting the program.
def main():
    
    g = getinput()

    print("Here is the group presentation corresponding to your input: \n", str(g))
    print("The rank of this group is: ", g.rank())
    print("Its Jacobi Matrix is given by: \n", np.array(g.jacobimatrix()))

    if g.rank() == 0:
        print("For a group of rank 0 we can not produce a non-trivial homomorphism to a free abelian group. In particular there is no Alexander function.")
    else:
        print("We can generate a group epimorphism onto its abelianization. It sends the generators to: \n", g.canonabhom())
        print("The induced Jacobi under this homomorphism looks like: \n", np.array(inducedjacobimatrix(g,g.canonabhom())))
        print("From this we can compute the Alexander function (numerator,denominator): \n", alexanderfunction(g,g.canonabhom()))

        if g.rank() == 1:
            print("Since the rank of this group is 1 there are only 2 epimorphisms onto its abelianisation. The other one is given by: \n", -g.canonabhom())
            print("The induced Jacobi under this homomorphism looks like: \n", np.array(inducedjacobimatrix(g,-g.canonabhom())))
            print("From this we can compute the Alexander function (numerator,denominator): \n", alexanderfunction(g,-g.canonabhom()))

        showplot= input("Would you like to see the coeffiecients of the associated power series in a plot? (y/n) \n")
        if showplot == "y":
            N = input("Please type in a natural number for the coefficient of the highest degree you want to see \n")
            if int(N) >= 0:
                plotalexanderfunction(g,g.canonabhom(),int(N))
                if g.rank() == 1:
                    plotalexanderfunction(g,-g.canonabhom(),int(N))
            else: 
                raise RuntimeError("This is not a natural number. Please restart the program.")

main()