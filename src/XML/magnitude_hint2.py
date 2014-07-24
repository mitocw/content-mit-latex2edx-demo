# the global variable "expected" should be defined before functions in this script are called
#
# if the problem has more than one answer box, then "expected" should be a list, and
# hint_mag0, hint_mag1, etc should be used in sequential order for each answer box.

from math import log10
from calc import evaluator

def is_close(ans, expect):
    # compute difference of ans and expected, in terms of order of magnitude
    try:
        magdif = abs( log10(abs(float(ans)))-log10(abs(expect)) )
    except Exception as err:
        return ""
    if magdif > 2:
        return "You are more than two orders of magnitude off&lt;br/&gt;"
    return ""

def is_tight(ans, expect, tolpct=0.01):
    # compute difference of ans and expected, and return True if within tolerance percentage
    difpct = abs( float(ans) - float(expect) ) / float(expect)
    return difpct <= tolpct

def in_range(ans, range):
    ansf = float(ans)
    return ansf >= range[0] and ansf <= range[1]

def is_sign_correct(ans, expect):
    # see if sign is same between ans and expect
    try:
        signdif = math.copysign(1, float(ans)) * math.copysign(1, float(expect))
    except Exception as err:
        return ""
    if signdif < 0:
        return "Is the sign of your answer correct?&lt;br/&gt;"
    return ""

def hint_mag0_sign(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=0, sign=True)

def hint_mag0(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=0)

def hint_mag1_sign(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=1, sign=True)

def hint_mag1(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=1)

def hint_mag2_sign(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=2, sign=True)

def hint_mag2(answer_ids, student_answers, new_cmap, old_cmap):
    hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=2)

def hint_mag(answer_ids, student_answers, new_cmap, old_cmap, anum=0, sign=False):
    global expected

    try:
        aid = answer_ids[0]
    except Exception as err:
        raise Exception('cannot get answer_ids[%d], answer_ids=%s, new_cmap=%s, err=%s' % (anum, answer_ids, new_cmap, err))

    ans = student_answers[aid]
    try:
        ans = float(ans)
    except Exception as err:
        try:
            ans = evaluator({},{}, ans)
        except Exception as err:
            hint = '<font color="red">Cannot evaluate your answer</font>'
            new_cmap.set_hint_and_mode(aid, hint, 'always')
            return

    try:
        if type(expected)==list:
            expect = expected[anum]
        else:
            expect = expected
    except Exception as err:
        raise Exception('expected answer not evaluated, expected=%s, anum=%s, err=%s' % (expected, anum, str(err)))

    # if expect is a dict, then generate hints by range in addition to
    extra_hints = []
    hint = ''
    if type(expect)==dict:
        expect_dict = expect
        expect = expect_dict['val']
        extra_hints = expect_dict.get('extra_hints', [])

    if new_cmap.is_correct(aid):
        # if correct, make sure answer is close, else direct student to look at solution
        if not is_tight(ans, expect, 0.01):
            hint = '<font color="green">Your answer is accepted as correct, but more than 1% from the expected.  Please check the solutions, and use the expected answer in any further calculations.</font>'

    else:
        hint = is_close(ans, expect)
        if not hint and sign:
            hint = is_sign_correct(ans, expect)

        for eh in extra_hints:
            range = eh.get('range','')
            if range:
                if in_range(ans, range):
                    hint += '  ' + eh['hint']

    if hint:
        new_cmap.set_hint_and_mode(aid, hint, 'always')
