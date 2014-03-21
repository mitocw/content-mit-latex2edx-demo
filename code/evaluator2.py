#
# compare_with_tolerance
# is_formula_equal
#

import numpy
import numbers
import random
from math import *
from calc2 import evaluator

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
