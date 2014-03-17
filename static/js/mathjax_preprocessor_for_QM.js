$('.text-input-dynamath_data').addClass('inline');	// tmp fix until https://github.com/edx/edx-platform/pull/2869 gets merged

window.MathjaxPreprocessorForQM = function () {
    /*----------------------------------------------------------------------
     * Translate conventions for quantum mechanics math for MathJax
     *   '>' --> '\rangle'
     *----------------------------------------------------------------------
     */
    this.fn = function (eqn) {
        // Default keywords are taken from capa/calc.py
        var default_keywords = ['sin', 'cos', 'tan', 'sqrt', 'log10', 'log2', 'ln', 'arccos', 'arcsin', 'arctan', 'abs', 'pi', 'inf'];

        // Escape keywords are strings that have special meaning in QM that should not be processed by MathJax
        var escape_keywords = ['ket'];

        // handle ket
        var replace_ket = function (match) {
            return '\\rangle:}';
        };
        eqn = eqn.replace(/>/g, replace_ket);

	// handle vertical bar, to add an invisible leftbracket (see common/static/js/vendor/mathjax-MathJax-c9db6ac/unpacked/jax/input/AsciiMath/jax.js)
        var replace_vert = function (match) {
            return ':{:';
        };
        eqn = eqn.replace(/\|/g, replace_vert);

        var replace_vert2 = function (match) {
            return '{:|{:';
        };
        eqn = eqn.replace(/:{:/g, replace_vert2);

        // handle bra
        var replace_bra = function (match) {
            return '\\langle';
        };
        eqn = eqn.replace(/</g, replace_bra);

        // Second, escape QM specific keywords from MathJax
        var replace_escape_keyword = function (match) {
            return '"' + match + '"'; // Force MathJax plain text
        };
        for(i=0; i<escape_keywords.length; i++) {
            var escape_keyword = escape_keywords[i];
            var escape_pattern = RegExp('\\b'+escape_keyword+'\\b','gi');
            eqn = eqn.replace(escape_pattern, replace_escape_keyword);
        }

        return eqn;
    };
}
