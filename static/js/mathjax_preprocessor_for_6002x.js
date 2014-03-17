window.MathjaxPreprocessorFor6002x = function () {
    /*----------------------------------------------------------------------
     * Translate 6.002x subscript convention to standard Latex, e.g.
     *   'R3'  --> 'R_{3}'
     *   'vGS' --> 'v_{GS}'
     *   'K/2*(vIN-VT)^2' --> 'K/2*(v_{IN}-V_{T})^2'
     * and also escape 6.002x-specific keywords from MathJax, such 
     *   as 'in' (parsed by MathJax as the set symbol)
     *----------------------------------------------------------------------
     */
    this.fn = function (eqn) {
        // Default keywords are taken from capa/calc.py
        var default_keywords = ['sin', 'cos', 'tan', 'sqrt', 'log10', 'log2', 'ln', 'arccos', 'arcsin', 'arctan', 'abs', 'pi', 'inf'];

        // Escape keywords are strings that have special meaning in 6.002x that should not be processed by MathJax
        var escape_keywords = ['in', 'out'];

        // Some keywords we will do substitutions
        var substitutions = { 'w':'omega', 'a': 'alpha', 'inf': 'infty' };

        // Zeroth, catch exponentiation
        var replace_exponentiation = function (match) {
            return '^{' + match.substr(1) + '}';
        };
        eqn = eqn.replace(/[\^]\w+/g, replace_exponentiation);

        // First, perform subscript insertion, but watch out for keywords
        var replace_subscript = function (match) {
            if ((default_keywords.indexOf(match) >= 0) || (escape_keywords.indexOf(match) >= 0))
                return match;
            return match.charAt(0) + '_{' + match.substr(1) + '}';
        };
        eqn = eqn.replace(/[A-Za-z]\w+/g, replace_subscript);

        // Second, escape 6.002x specific keywords from MathJax
        var replace_escape_keyword = function (match) {
            return '"' + match + '"'; // Force MathJax plain text
        };
        for(i=0; i<escape_keywords.length; i++) {
            var escape_keyword = escape_keywords[i];
            var escape_pattern = RegExp('\\b'+escape_keyword+'\\b','gi');
            eqn = eqn.replace(escape_pattern, replace_escape_keyword);
        }

        // Third, do course-specific substitutions
        var replace_substitute = function (match) {
            if (match[match.length-1]=='_')
                return substitutions[match.slice(0,-1)]+'_';
            else
                return substitutions[match];
        };
        for(var substitution in substitutions) {
            var substitution_pattern = RegExp('\\b'+substitution+'_*\\b','g');
            eqn = eqn.replace(substitution_pattern, replace_substitute);
        }
        return eqn;
    };
}
