#
# formula_evaluator: allows multiple possible answers, using options
#

import numpy
from evaluator2 import is_formula_equal

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

def test():
    x = numpy.matrix('[1,2;3,4]')
    y = numpy.matrix('[4,9;2,7]')
    samples="x,y@[1|2;3|4],[0|2;4|6]:[5|5;5|5],[8|8;8|8]#50"

    print "test_formula gives %s" %  test_formula('x*y', 'x*y', options="samples='x,y,i@[1|2;3|4],[0|2;4|6],0+1j:[5|5;5|5],[8|8;8|8],0+1j#50'!altanswer='-y*x'")

    return x,y, samples

