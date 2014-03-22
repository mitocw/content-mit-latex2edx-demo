#
# all evaluators and calc2 in one file - must use this on
# edX studio because it doesn't support use of /python_lib
# or /code

#-----------------------------------------------------------------------------
# calc2.py

"""
Parser and evaluator for FormulaResponse and NumericalResponse

Uses pyparsing to parse. Main function as of now is evaluator().

This version handles matrices, via numpy
"""

import math
import operator
import numbers
import numpy
import scipy.constants
from calc import functions

from pyparsing import (
    Word, Literal, CaselessLiteral, ZeroOrMore, MatchFirst, Optional, Forward,
    Group, ParseResults, stringEnd, Suppress, Combine, alphas, nums, alphanums
)

DEFAULT_FUNCTIONS = {
    'sin': numpy.sin,
    'cos': numpy.cos,
    'tan': numpy.tan,
    'sec': functions.sec,
    'csc': functions.csc,
    'cot': functions.cot,
    'sqrt': numpy.sqrt,
    'log10': numpy.log10,
    'log2': numpy.log2,
    'ln': numpy.log,
    'exp': numpy.exp,
    'arccos': numpy.arccos,
    'arcsin': numpy.arcsin,
    'arctan': numpy.arctan,
    'arcsec': functions.arcsec,
    'arccsc': functions.arccsc,
    'arccot': functions.arccot,
    'abs': numpy.abs,
    'fact': math.factorial,
    'factorial': math.factorial,
    'sinh': numpy.sinh,
    'cosh': numpy.cosh,
    'tanh': numpy.tanh,
    'sech': functions.sech,
    'csch': functions.csch,
    'coth': functions.coth,
    'arcsinh': numpy.arcsinh,
    'arccosh': numpy.arccosh,
    'arctanh': numpy.arctanh,
    'arcsech': functions.arcsech,
    'arccsch': functions.arccsch,
    'arccoth': functions.arccoth
}
DEFAULT_VARIABLES = {
    'i': numpy.complex(0, 1),
    'j': numpy.complex(0, 1),
    'e': numpy.e,
    'pi': numpy.pi,
    'k': scipy.constants.k,  # Boltzmann: 1.3806488e-23 (Joules/Kelvin)
    'c': scipy.constants.c,  # Light Speed: 2.998e8 (m/s)
    'T': 298.15,  # Typical room temperature: 298.15 (Kelvin), same as 25C/77F
    'q': scipy.constants.e  # Fund. Charge: 1.602176565e-19 (Coulombs)
}

# We eliminated the following extreme suffixes:
#   P (1e15), E (1e18), Z (1e21), Y (1e24),
#   f (1e-15), a (1e-18), z (1e-21), y (1e-24)
# since they're rarely used, and potentially confusing.
# They may also conflict with variables if we ever allow e.g.
#   5R instead of 5*R
SUFFIXES = {
    '%': 0.01, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
    'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12
}


class UndefinedVariable(Exception):
    """
    Indicate when a student inputs a variable which was not expected.
    """
    pass


def lower_dict(input_dict):
    """
    Convert all keys in a dictionary to lowercase; keep their original values.

    Keep in mind that it is possible (but not useful?) to define different
    variables that have the same lowercase representation. It would be hard to
    tell which is used in the final dict and which isn't.
    """
    return {k.lower(): v for k, v in input_dict.iteritems()}


# The following few functions define evaluation actions, which are run on lists
# of results from each parse component. They convert the strings and (previously
# calculated) numbers into the number that component represents.

def super_float(text):
    """
    Like float, but with SI extensions. 1k goes to 1000.
    """
    if text[-1] in SUFFIXES:
        return float(text[:-1]) * SUFFIXES[text[-1]]
    else:
        return float(text)


def eval_number(parse_result):
    """
    Create a float out of its string parts.

    e.g. [ '7.13', 'e', '3' ] ->  7130
    Calls super_float above.
    """
    return super_float("".join(parse_result))


def is_final(elem):
    if isinstance(elem, numbers.Number):
        return True
    if isinstance(elem, numpy.matrix):
        return True


def eval_atom(parse_result):
    """
    Return the value wrapped by the atom.

    In the case of parenthesis, ignore them.
    """
    # Find first number in the list
    result = next(k for k in parse_result if is_final(k))
    return result


def eval_power(parse_result):
    """
    Take a list of numbers and exponentiate them, right to left.

    e.g. [ 2, 3, 2 ] -> 2^3^2 = 2^(3^2) -> 512
    (not to be interpreted (2^3)^2 = 64)
    """
    # `reduce` will go from left to right; reverse the list.
    parse_result = reversed(
        [k for k in parse_result
         if is_final(k)]  # Ignore the '^' marks.
    )
    # Having reversed it, raise `b` to the power of `a`.
    power = reduce(lambda a, b: b ** a, parse_result)
    return power


def eval_parallel(parse_result):
    """
    Compute numbers according to the parallel resistors operator.

    BTW it is commutative. Its formula is given by
      out = 1 / (1/in1 + 1/in2 + ...)
    e.g. [ 1, 2 ] -> 2/3

    Return NaN if there is a zero among the inputs.
    """
    if len(parse_result) == 1:
        return parse_result[0]
    if 0 in parse_result:
        return float('nan')
    reciprocals = [1. / e for e in parse_result
                   if isinstance(e, numbers.Number)]
    return 1. / sum(reciprocals)


def eval_sum(parse_result):
    """
    Add the inputs, keeping in mind their sign.

    [ 1, '+', 2, '-', 3 ] -> 0

    Allow a leading + or -.
    """
    total = 0.0
    current_op = operator.add
    for token in parse_result:
        if token == '+':
            current_op = operator.add
        elif token == '-':
            current_op = operator.sub
        else:
            total = current_op(total, token)
    return total


def eval_product(parse_result):
    """
    Multiply the inputs.

    [ 1, '*', 2, '/', 3 ] -> 0.66
    """
    prod = 1.0
    current_op = operator.mul
    for token in parse_result:
        if token == '*':
            current_op = operator.mul
        elif token == '/':
            current_op = operator.truediv
        else:
            prod = current_op(prod, token)
    return prod


def add_defaults(variables, functions, case_sensitive):
    """
    Create dictionaries with both the default and user-defined variables.
    """
    all_variables = dict(DEFAULT_VARIABLES)
    all_functions = dict(DEFAULT_FUNCTIONS)
    all_variables.update(variables)
    all_functions.update(functions)

    if not case_sensitive:
        all_variables = lower_dict(all_variables)
        all_functions = lower_dict(all_functions)

    return (all_variables, all_functions)


def evaluator(variables, functions, math_expr, case_sensitive=False):
    """
    Evaluate an expression; that is, take a string of math and return a float.

    -Variables are passed as a dictionary from string to value. They must be
     python numbers.
    -Unary functions are passed as a dictionary from string to function.
    """
    # No need to go further.
    if math_expr.strip() == "":
        return float('nan')

    # Parse the tree.
    math_interpreter = ParseAugmenter(math_expr, case_sensitive)
    math_interpreter.parse_algebra()

    # Get our variables together.
    all_variables, all_functions = add_defaults(variables, functions, case_sensitive)

    # ...and check them
    math_interpreter.check_variables(all_variables, all_functions)

    # Create a recursion to evaluate the tree.
    if case_sensitive:
        casify = lambda x: x
    else:
        casify = lambda x: x.lower()  # Lowercase for case insens.

    evaluate_actions = {
        'number': eval_number,
        'variable': lambda x: all_variables[casify(x[0])],
        'function': lambda x: all_functions[casify(x[0])](x[1]),
        'atom': eval_atom,
        'power': eval_power,
        'parallel': eval_parallel,
        'product': eval_product,
        'sum': eval_sum
    }

    return math_interpreter.reduce_tree(evaluate_actions)


class ParseAugmenter(object):
    """
    Holds the data for a particular parse.

    Retains the `math_expr` and `case_sensitive` so they needn't be passed
    around method to method.
    Eventually holds the parse tree and sets of variables as well.
    """
    def __init__(self, math_expr, case_sensitive=False):
        """
        Create the ParseAugmenter for a given math expression string.

        Do the parsing later, when called like `OBJ.parse_algebra()`.
        """
        self.case_sensitive = case_sensitive
        self.math_expr = math_expr
        self.tree = None
        self.variables_used = set()
        self.functions_used = set()

        def vpa(tokens):
            """
            When a variable is recognized, store it in `variables_used`.
            """
            varname = tokens[0][0]
            self.variables_used.add(varname)

        def fpa(tokens):
            """
            When a function is recognized, store it in `functions_used`.
            """
            varname = tokens[0][0]
            self.functions_used.add(varname)

        self.variable_parse_action = vpa
        self.function_parse_action = fpa

    def parse_algebra(self):
        """
        Parse an algebraic expression into a tree.

        Store a `pyparsing.ParseResult` in `self.tree` with proper groupings to
        reflect parenthesis and order of operations. Leave all operators in the
        tree and do not parse any strings of numbers into their float versions.

        Adding the groups and result names makes the `repr()` of the result
        really gross. For debugging, use something like
          print OBJ.tree.asXML()
        """
        # 0.33 or 7 or .34 or 16.
        number_part = Word(nums)
        inner_number = (number_part + Optional("." + Optional(number_part))) | ("." + number_part)
        # pyparsing allows spaces between tokens--`Combine` prevents that.
        inner_number = Combine(inner_number)

        # SI suffixes and percent.
        number_suffix = MatchFirst(Literal(k) for k in SUFFIXES.keys())

        # 0.33k or 17
        plus_minus = Literal('+') | Literal('-')
        number = Group(
            Optional(plus_minus) +
            inner_number +
            Optional(CaselessLiteral("E") + Optional(plus_minus) + number_part) +
            Optional(number_suffix)
        )
        number = number("number")

        # Predefine recursive variables.
        expr = Forward()

        # Handle variables passed in. They must start with letters/underscores
        # and may contain numbers afterward.
        inner_varname = Word(alphas + "_", alphanums + "_")
        varname = Group(inner_varname)("variable")
        varname.setParseAction(self.variable_parse_action)

        # Same thing for functions.
        function = Group(inner_varname + Suppress("(") + expr + Suppress(")"))("function")
        function.setParseAction(self.function_parse_action)

        atom = number | function | varname | "(" + expr + ")"
        atom = Group(atom)("atom")

        # Do the following in the correct order to preserve order of operation.
        pow_term = atom + ZeroOrMore("^" + atom)
        pow_term = Group(pow_term)("power")

        par_term = pow_term + ZeroOrMore('||' + pow_term)  # 5k || 4k
        par_term = Group(par_term)("parallel")

        prod_term = par_term + ZeroOrMore((Literal('*') | Literal('/')) + par_term)  # 7 * 5 / 4
        prod_term = Group(prod_term)("product")

        sum_term = Optional(plus_minus) + prod_term + ZeroOrMore(plus_minus + prod_term)  # -5 + 4 - 3
        sum_term = Group(sum_term)("sum")

        # Finish the recursion.
        expr << sum_term  # pylint: disable=W0104
        self.tree = (expr + stringEnd).parseString(self.math_expr)[0]

    def reduce_tree(self, handle_actions, terminal_converter=None):
        """
        Call `handle_actions` recursively on `self.tree` and return result.

        `handle_actions` is a dictionary of node names (e.g. 'product', 'sum',
        etc&) to functions. These functions are of the following form:
         -input: a list of processed child nodes. If it includes any terminal
          nodes in the list, they will be given as their processed forms also.
         -output: whatever to be passed to the level higher, and what to
          return for the final node.
        `terminal_converter` is a function that takes in a token and returns a
        processed form. The default of `None` just leaves them as strings.
        """
        def handle_node(node):
            """
            Return the result representing the node, using recursion.

            Call the appropriate `handle_action` for this node. As its inputs,
            feed it the output of `handle_node` for each child node.
            """
            if not isinstance(node, ParseResults):
                # Then treat it as a terminal node.
                if terminal_converter is None:
                    return node
                else:
                    return terminal_converter(node)

            node_name = node.getName()
            if node_name not in handle_actions:  # pragma: no cover
                raise Exception(u"Unknown branch name '{}'".format(node_name))

            action = handle_actions[node_name]
            handled_kids = [handle_node(k) for k in node]
            return action(handled_kids)

        # Find the value of the entire tree.
        return handle_node(self.tree)

    def check_variables(self, valid_variables, valid_functions):
        """
        Confirm that all the variables used in the tree are valid/defined.

        Otherwise, raise an UndefinedVariable containing all bad variables.
        """
        if self.case_sensitive:
            casify = lambda x: x
        else:
            casify = lambda x: x.lower()  # Lowercase for case insens.

        # Test if casify(X) is valid, but return the actual bad input (i.e. X)
        bad_vars = set(var for var in self.variables_used
                       if casify(var) not in valid_variables)
        bad_vars.update(func for func in self.functions_used
                        if casify(func) not in valid_functions)

        if bad_vars:
            raise UndefinedVariable(' '.join(sorted(bad_vars)))

#-----------------------------------------------------------------------------
# evaluator2.py

#
# compare_with_tolerance
# is_formula_equal
#

import numpy
import numbers
import random
from math import *

default_tolerance = '0.01%'

def compare_with_tolerance(complex1, complex2, tolerance=default_tolerance, relative_tolerance=False):
    """
    Compare complex1 to complex2 with maximum tolerance tol.

    If tolerance is type string, then it is counted as relative if it ends in %; otherwise, it is absolute.

     - complex1    :  student result (float complex number)
     - complex2    :  instructor result (float complex number)
     - tolerance   :  string representing a number or float
     - relative_tolerance: bool, used when`tolerance` is float to explicitly use passed tolerance as relative.

     Default tolerance of 1e-3% is added to compare two floats for
     near-equality (to handle machine representation errors).
     Default tolerance is relative, as the acceptable difference between two
     floats depends on the magnitude of the floats.
     (http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
     Examples:
        In [183]: 0.000016 - 1.6*10**-5
        Out[183]: -3.3881317890172014e-21
        In [212]: 1.9e24 - 1.9*10**24
        Out[212]: 268435456.0
    """
    def myabs(elem):
        if isinstance(elem, numpy.matrix):
            return numpy.sum(abs(elem))
        return abs(elem)

    if isinstance(tolerance, numbers.Number):
        tolerance = str(tolerance)
    if relative_tolerance:
        tolerance = tolerance * max(myabs(complex1), myabs(complex2))
    elif tolerance.endswith('%'):
        tolerance = evaluator(dict(), dict(), tolerance[:-1]) * 0.01
        tolerance = tolerance * max(myabs(complex1), myabs(complex2))
    else:
        tolerance = evaluator(dict(), dict(), tolerance)

    try:
        if numpy.isinf(complex1).any() or numpy.isinf(complex2).any():
            # If an input is infinite, we can end up with `abs(complex1-complex2)` and
            # `tolerance` both equal to infinity. Then, below we would have
            # `inf <= inf` which is a fail. Instead, compare directly.
            cmp = (complex1 == complex2)
            if isinstance(cmp, numpy.matrix):
                return cmp.all()
            return cmp
        else:
            # v1 and v2 are, in general, complex numbers:
            # there are some notes about backward compatibility issue: see responsetypes.get_staff_ans()).
            # return abs(complex1 - complex2) <= tolerance
            #
            # sum() used to handle matrix comparisons
            return numpy.sum(abs(complex1 - complex2)) <= tolerance
    except Exception as err:
        print "failure in comparison, complex1=%s, complex2=%s" % (complex1, complex2)
        print "err = ", err
        raise


def is_formula_equal(expected, given, samples, cs=True, tolerance='0.01', evalfun=None, cmpfun=None, debug=False):
    '''
    samples examples:

    samples="m_I,m_J,I_z,J_z@1,1,1,1:20,20,20,20#50" 
    samples="J,m,Delta,a,h,x,mu_0,g_I,B_z@0.5,1,1,1,1,1,1,1,1:0.5,20,20,20,20,20,20,20,20#50"

    matrix sampling:

    samples="x,y@[1|2;3|4],[0|2;4|6]:[5|5;5|5],[8|8;8|8]#50"

    complex numbers:

    'x,y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'

    '''
    if evalfun is None:
        evalfun = evaluator
        #evalfun = EvaluateWithKets
    if cmpfun is None:
        def cmpfun(a, b, tol):
            return compare_with_tolerance(a, b, tol)
        
    variables = samples.split('@')[0].split(',')
    numsamples = int(samples.split('@')[1].split('#')[1])

    def to_math_atom(sstr):
        '''
        Convert sample range atom to float or to matrix
        '''
        if '[' in sstr:
            return numpy.matrix(sstr.replace('|',' '))
        elif 'j' in sstr:
            return complex(sstr)
        else:
            return float(sstr)

    sranges = zip(*map(lambda x: map(to_math_atom, x.split(",")),
                       samples.split('@')[1].split('#')[0].split(':')))
    ranges = dict(zip(variables, sranges))

    if debug:
        print "ranges = ", ranges

    for i in range(numsamples):
        vvariables = {}
        for var in ranges:
            value = random.uniform(*ranges[var])
            vvariables[str(var)] = value
        if debug:
            print "vvariables = ", vvariables
        try:
            instructor_result = evalfun(vvariables, dict(), expected, case_sensitive=cs)
        except Exception as err:
            #raise Exception("is_formula_eq: vvariables=%s, err=%s" % (vvariables, str(err)))
            #raise Exception("-- %s " % str(err))
            raise Exception("Error evaluating instructor result, expected=%s, vv=%s -- %s " % (expected, vvariables, str(err)))
        try:
            student_result = evalfun(vvariables, dict(), given, case_sensitive=cs)
        except Exception as err:
            #raise Exception("is_formula_eq: vvariables=%s, err=%s" % (vvariables, str(err)))
            raise Exception("-- %s " % str(err))
            # raise Exception("Error evaluating your input, given=%s, vv=%s -- %s " % (given, vvariables, str(err)))
        #print "instructor=%s, student=%s" % (instructor_result, student_result)
        cfret = cmpfun(instructor_result, student_result, tolerance)
        if debug:
            print "comparison result = %s" % cfret
        if not cfret:
            return False
    return True


#-----------------------------------------------------------------------------
# matrix_evaluator.py

#
# formula_evaluator: allows multiple possible answers, using options
#

import numpy
#from evaluator2 import is_formula_equal

def test_formula(expect, ans, options=None):
    '''
    expect and ans are math expression strings.
    Check for equality using random sampling.
    options should be like samples="m_I,m_J,I_z,J_z@1,1,1,1:20,20,20,20#50"!tolerance=0.3
    i.e. a sampling range for the equality testing, in the same
    format as used in formularesponse.

    options may also include altanswer, an alternate acceptable answer.  Example:

    options="samples='X,Y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'!altanswer='-Y*X'"

    note that the different parts of the options string are to be spearated by a bang (!).
    '''
    samples = None
    tolerance = '0.1%'
    acceptable_answers = [expect]
    for optstr in options.split('!'):
        if 'samples=' in optstr:
            samples = eval(optstr.split('samples=')[1])
        elif 'tolerance=' in optstr:
            tolerance = eval(optstr.split('tolerance=')[1])
        elif 'altanswer=' in optstr:
            altanswer = eval(optstr.split('altanswer=')[1])
            acceptable_answers.append(altanswer)
    if samples is None:
        return {'ok': False, 'msg': 'Oops, problem authoring error, samples=None'}

    # for debugging
    # return {'ok': False, 'msg': 'altanswer=%s' % altanswer}

    for acceptable in acceptable_answers:
        try:
            ok = is_formula_equal(acceptable, ans, samples, cs=True, tolerance=tolerance)
        except Exception as err:
            return {'ok': False, 'msg': "Sorry, could not evaluate your expression.  Error %s" % str(err)}
        if ok:
            return {'ok':ok, 'msg': ''}

    return {'ok':ok, 'msg': ''}

def matrix_test():
    x = numpy.matrix('[1,2;3,4]')
    y = numpy.matrix('[4,9;2,7]')
    samples="x,y@[1|2;3|4],[0|2;4|6]:[5|5;5|5],[8|8;8|8]#50"

    print "test_formula gives %s" %  test_formula('x*y', 'x*y', options="samples='x,y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'!altanswer='-y*x'")

    return x,y, samples

#-----------------------------------------------------------------------------
# ket_evaluator.py

#
# ket_evaluator: compare QM ket formulas
#

import re
#import numpy
#from calc2 import evaluator

# evaluate expression with |...> kets inside
# return a VectorState

class VectorState(object):
    '''
    Representation of a vector in terms of a list of pairs.
    Each pair has (vector_label, coefficient) where

    vector_label = list of numbers labeling the vector
    coefficient = weight of that vector

    Each vector label should be unique.

    Equality checking is supported.  Two VectorState
    instances are equal if and only if they have 
    the same number of pairs, and the sets of
    vector labels are equal, and the coefficients
    of each vector label between the two VectorStae
    instances are equal.
    '''
    def __init__(self, pairs):
        '''
        pairs = dict of label, coefficient
        '''
        self.pairs = pairs

    def labels(self):
        return set(self.pairs.keys())

    def __str__(self):
        return str(self.pairs)

    def __repr__(self):
        return repr(self.pairs) 

    def __eq__(self, other):
        '''
        Check for equality.  Return True if equal.
        '''
        if self.labels() != other.labels():
            return False
        for label in self.labels():
            if self.pairs[label] != other.pairs[label]:
                return False
        return True

    def eq_with_tolerance(self, other, tolerance):
        '''
        Check for closeness, within tolerance.
        '''
        if self.labels() != other.labels():
            return False
        for label in self.labels():
            if not compare_with_tolerance(self.pairs[label], other.pairs[label], tolerance):
                return False
        return True
            

class KetVector(object):
    '''
    Representation of a |label> ket object.
    '''
    def __init__(self, kvstring):
        self.kvstring = kvstring

    def evaluator(self, variables, functions, case_sensitive=False):
       exprs = self.kvstring[1:-1].split(',')
       self.label = tuple([ evaluator(variables, functions, x, case_sensitive) for x in exprs ])
       return self.label

    def has_label(self, olb):
        '''
        Return True if olb is close to self.label
        '''
        return self.label == olb

    def __str__(self):
        return self.kvstring

    __repr__ = __str__

def EvaluateWithKets(variables, functions, math_expr, case_sensitive=False):
    '''
    Evaluate expression with quantum mechanical "kets" in the expression.
    Each "ket" is of the form |...> and represents a unit vector.
    We assume the kets are either orthogonal or equal to each other:
    orthogonal when the labels are different, and equal when the labels
    are equal.
    '''
    
    # avoid all XML encoding issues by not having greater-than sign or ampersand in textr
    gtch = chr(62)
    ampch = chr(38)
    math_expr = math_expr.replace(ampch + 'gt;', gtch)

    # extract list of all kets
    kvlist = map(KetVector, re.findall('\|[^{0}|]+{0}'.format(gtch), math_expr))
    kvdict = {kv.kvstring: kv for kv in kvlist}

    mehex = str(map(ord, math_expr))

    # for debugging
    if 0:
        raise Exception('type=%s, vcnt=%d, gcnt=%d, len=%d, math_expr=%s, kvlist=%s' % (type(math_expr),
                                                                                        math_expr.count('|'),
                                                                                        math_expr.count(gtch),
                                                                                        
                                                                                        len(kvlist), 
                                                                                        mehex,
                                                                                        str(kvlist).replace('>','&gt;')))
        raise Exception('kvdict=%s' % str(kvdict).replace('>','&gt;'))

    if not kvlist:
        return VectorState({None: evaluator(variables, functions, math_expr, case_sensitive)})

    # get set of unique kets, after evaluating their labels
    try:
        kvlabels = set([ kv.evaluator(variables, functions, case_sensitive) for kv in kvlist ])
    except Exception as err:
        raise Exception('Cannot evaluate ket label, err=%s' % err)
    
    # raise Exception('labels=%s' % kvlabels)

    # for each unique label, get coefficient
    # do this by setting all but that ket to zero and evaulating the whole math expression
    coefficient = {}
    for label in kvlabels:
        
        def kvsub(match):
            kv = kvdict[match.group(1)]
            if kv.has_label(label):
                return '1'
            return '0'
        new_expr = re.sub('(\|[^>|]+>)', kvsub, math_expr)

        # print "    label=%s, new_expr=%s" % (label, new_expr)
        coefficient[label] = evaluator(variables, functions, new_expr, case_sensitive)
    vs = VectorState(coefficient)
    # print "%s -> %s" % (math_expr, vs)
    return vs
    
def test_ket(expect, ans, options=None):
    '''
    expect and ans are math expression strings which may include quantum-mechanics "ket"s.
    Check for equality using random sampling via EvaluateWithKets.
    options should be like samples="m_I,m_J,I_z,J_z@1,1,1,1:20,20,20,20#50"|tolerance=0.3
    i.e. a sampling range for the equality testing, in the same
    format as used in formularesponse.
    '''
    samples = None
    tolerance = '0.1%'
    acceptable_answers = [expect]
    for optstr in options.split('!'):
        if 'samples=' in optstr:
            samples = eval(optstr.split('samples=')[1])
        elif 'tolerance=' in optstr:
            tolerance = eval(optstr.split('tolerance=')[1])
        elif 'altanswer=' in optstr:
            altanswer = eval(optstr.split('altanswer=')[1])
            acceptable_answers.append(altanswer)
    if samples is None:
        return {'ok': False, 'msg': 'Oops, problem authoring error, samples=None'}

    def cmpfun(a, b, tol):
        return a.eq_with_tolerance(b, tol)
    
    for acceptable in acceptable_answers:
        try:
            ok = is_formula_equal(acceptable, ans, samples, cs=True, tolerance=tolerance, evalfun=EvaluateWithKets, cmpfun=cmpfun)
        except Exception as err:
            return {'ok': False, 'msg': "Sorry, could not evaluate your expression.  Error %s" % str(err)}
        if ok:
            return {'ok':ok, 'msg': ''}

    return {'ok':ok, 'msg': ''}

def ket_evaluator_test():
    expect = "|0>+3*|b>"
    ans = "(|0>+2*|b>+|b>)*1.0"
    options="samples='b@1:10#20'"
    print "test_ket returns %s" % test_ket(expect, ans, options=options)

