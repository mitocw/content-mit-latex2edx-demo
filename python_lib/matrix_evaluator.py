#
# formula_evaluator: allows multiple possible answers, using options
#

import numpy
import scipy
from calc2 import evaluator
from evaluator2 import is_formula_equal

def mfun(linalgfun):
    '''
    scipy.linalg functions return arrays instead of numpy matrix objects.
    force objects to be matrices, so that the evaluator's is_final function
    knows when to stop.  We consistently keep objects as being numbers
    or matrices, and not arrays.
    '''
    return lambda x: numpy.matrix(linalgfun(x))


def matrix_evaluator(variables, functions, math_expr, case_sensitive=False):
    '''
    Do same as normal evaluator, but override some functions with ones which
    can handle matrices, like expm for matrix exponentiation.
    '''
    #mfunctions = {'exp': mfun(scipy.linalg.expm3),
    mfunctions = {'exp': mfun(scipy.linalg.expm),
                  'cos': mfun(scipy.linalg.cosm),
                  'sin': mfun(scipy.linalg.sinm),
                  'tan': mfun(scipy.linalg.tanm),
                  'sqrt': mfun(scipy.linalg.sqrtm),
                 }
    return evaluator(variables, mfunctions, math_expr, case_sensitive)


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
            ok = is_formula_equal(acceptable, ans, samples, cs=True, tolerance=tolerance, evalfun=matrix_evaluator)
        except Exception as err:
            return {'ok': False, 'msg': "Sorry, could not evaluate your expression.  Error %s" % str(err)}
        if ok:
            return {'ok':ok, 'msg': ''}

    return {'ok':ok, 'msg': ''}

def test():
    x = numpy.matrix('[1,2;3,4]')
    y = numpy.matrix('[4,9;2,7]')
    samples="x,y@[1|2;3|4],[0|2;4|6]:[5|5;5|5],[8|8;8|8]#50"

    print "test_formula gives %s" %  test_formula('x*y', 'x*y', options="samples='x,y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'!altanswer='-y*x'")

    return x,y, samples

def test2():
    expect = "exp(i*H_0*t/hbar) * H * exp(-i*H_0*t/hbar) - H_0"
    ans2   = "exp(i*H_0*t/hbar) * (H-H_0) * exp(-i*H_0*t/hbar)"
    samples = "hbar,t,H_0,H@1,0.1,[1|2;3|4],[1|3;4|6]:1,2,[5|5;5|5],[3|3;4|4]#50"
    options = "samples='%s'!tolerance='2%%'" % samples
    ans_wrong = "exp(i*H_0*t/hbar) * exp(-i*H_0*t/hbar) * H - H_0"
    ans = "-H_0 + exp(i*H_0*t/hbar) * H * exp(-i*H_0*t/hbar)"
    print "ans_wrong gives %s" %  test_formula(expect,
                                                  ans_wrong,
                                                  options=options)
    print "ans gives %s" %  test_formula(expect,
                                                  ans,
                                                  options=options)
    print "ans2 gives %s" %  test_formula(expect,
                                                  ans2,
                                                  options=options)
    return expect, samples, ans, ans2

def test3():
    f1 = "exp(i*Y) * H * exp(-i*Y) + Y"
    f2 = "exp(i*Y) * (H+Y) * exp(-i*Y)"
    H = numpy.matrix('[1,2;3,4]')
    Y = numpy.matrix('[5,6;7,8]')
    x1 = matrix_evaluator({'H': H, 'Y': Y}, {}, f1)
    x2 = matrix_evaluator({'H': H, 'Y': Y}, {}, f2)
    print "f1 = ", x1
    print "f2 = ", x2
    return f1, f2, x1, x2, H, Y

def test4():
    expect = "exp(i*H_0*t/hbar) * H * exp(-i*H_0*t/hbar) - H_0"
    ans2   = "exp(i*H_0*t/hbar) * (H-H_0) * exp(-i*H_0*t/hbar)"
    H0 = numpy.matrix('[1,2;3,4]')
    H = numpy.matrix('[1,3;4,6]')

    H0 = numpy.matrix('[5,5;5,5]')
    H = numpy.matrix('[3,3;4,4]')
    vars = {'H': H, 'H_0': H0, 'hbar': 1, 't': 2}
    x1 = matrix_evaluator(vars, {}, expect)
    x2 = matrix_evaluator(vars, {}, ans2)
    print "expect = ", x1
    print "ans2 = ", x2
    return expect, ans2, H0, H, vars

def test5():
    expect = "exp(i*H_0*t/hbar) * H * exp(-i*H_0*t/hbar) - H_0"
    ans2   = "exp(i*H_0*t/hbar) * (H-H_0) * exp(-i*H_0*t/hbar)"
    samples = "hbar,t,H_0,H@1,1,[1|2;3|4],[1|3;4|6]:1,1,[1|2;3|4],[1|3;4|6]#2"
    options = "samples='%s'!tolerance='0.1'" % samples
    print "ans2 gives %s" %  test_formula(expect, ans2,options=options)
    
    
