import json
import math

def checkans(xok, statesok, x, states):
    ep = 0.18
    if abs(x-xok)>ep:
        return False
    if states[0] not in statesok:
        return False
    if states[1]==states[0]:
        return False
    if states[1] not in statesok:
        return False
    return True


def test_findep(expect, ans):
    # for debugging
    # return {'ok': False, 'msg': 'ans=%s' % ans}

    # ans is a JSON string with "answer" and "state" defined
    answer = json.loads(json.loads(ans)['answer'])
    
    # return {'ok': False, 'msg': 'a=%s' % answer}
    
    # the javascript gives us a dict with x, astate, and bstate
    # legal answers for this question are:

    # x~0: c,g (1,0)-(2,0)
    # x~0.5: d,f (1,-1)-(2,-1)
    # x~2-sqrt(3) = 0.2679491924311228: d,g (1,0)-(2,-1)
    # x~2-sqrt(3) = 0.2679491924311228: f,c (1,-1)-(2,0)

    x = float(answer['x'])
    astate = str(answer['astate'])
    bstate = str(answer['bstate'])

    valid = {0: 'CG',
             0.5: 'DF',
             0.267951: 'DG',
             0.26795: 'FC',
             }

    for xok, statesok in valid.items():
        if checkans(xok, statesok, x, [astate, bstate]):
            return {'ok': True, 'msg': 'Correct!'}
    msg = "Sorry, the transition between %s and %s at x=%6.2f is not first-order field independent" % (astate,bstate,x)
    return {'ok': False, 'msg': msg}
