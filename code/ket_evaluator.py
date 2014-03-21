#
# ket_evaluator: compare QM ket formulas
#

import re
import numpy
from evaluator2 import is_formula_equal, compare_with_tolerance
from calc2 import evaluator

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

def test():
    expect = "|0>+3*|b>"
    ans = "(|0>+2*|b>+|b>)*1.0"
    options="samples='b@1:10#20'"
    print "test_ket returns %s" % test_ket(expect, ans, options=options)

