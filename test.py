vowels = frozenset(["a", "e", "i", "o", "u"])
def is_consonant(word, i):
    """Returns True if word[i] is a consonant, False otherwise

    A consonant is defined in the paper as follows:

        A consonant in a word is a letter other than A, E, I, O or
        U, and other than Y preceded by a consonant. (The fact that
        the term `consonant' is defined to some extent in terms of
        itself does not make it ambiguous.) So in TOY the consonants
        are T and Y, and in SYZYGY they are S, Z and G. If a letter
        is not a consonant it is a vowel.
    """
    if word[i] in vowels:
        return False
    if word[i] == "y":
        if i == 0:
            return True
        else:
            return not is_consonant(word, i - 1)
    return True


def measure(stem):
    r"""Returns the 'measure' of stem, per definition in the paper

    From the paper:

        A consonant will be denoted by c, a vowel by v. A list
        ccc... of length greater than 0 will be denoted by C, and a
        list vvv... of length greater than 0 will be denoted by V.
        Any word, or part of a word, therefore has one of the four
        forms:

            CVCV ... C
            CVCV ... V
            VCVC ... C
            VCVC ... V

        These may all be represented by the single form

            [C]VCVC ... [V]

        where the square brackets denote arbitrary presence of their
        contents. Using (VC){m} to denote VC repeated m times, this
        may again be written as

            [C](VC){m}[V].

        m will be called the \measure\ of any word or word part when
        represented in this form. The case m = 0 covers the null
        word. Here are some examples:

            m=0    TR,  EE,  TREE,  Y,  BY.
            m=1    TROUBLE,  OATS,  TREES,  IVY.
            m=2    TROUBLES,  PRIVATE,  OATEN,  ORRERY.
    """
    cv_sequence = ""

    # Construct a string of 'c's and 'v's representing whether each
    # character in `stem` is a consonant or a vowel.
    # e.g. 'falafel' becomes 'cvcvcvc',
    #      'architecture' becomes 'vcccvcvccvcv'
    for i in range(len(stem)):
        if is_consonant(stem, i):
            cv_sequence += "c"
        else:
            cv_sequence += "v"

    # Count the number of 'vc' occurrences, which is equivalent to
    # the number of 'VC' occurrences in Porter's reduced form in the
    # docstring above, which is in turn equivalent to `m`
    print(cv_sequence)
    return cv_sequence.count("vc")

print(measure('feed'))