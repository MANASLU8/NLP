import unittest

from pytest import mark

from source.task1.tokenizer import tokenize


@mark.parametrize('text, tag, exists', [
    # PGP
    ("""-----BEGIN PGP SIGNATURE-----
Version: 2.2
iQBuAgUBK8DNazh0K1zBsGrxAQFoZQLEC/XKXMoDhCPf/AZ3aOQSLfz+6w400UDk
Ng6prxnPuEuSZQEiiusMCVcRcGnWbaVrxFjA1o4yubh01Czcg3ZC9wLJolXlxJn7
iSJh/eTZxmJnNynJxlGs0Ao=
=4eZb
-----END PGP SIGNATURE-----""", "PGP_SIG", True),
    ("""-----BEGIN PGP PUBLIC KEY BLOCK-----
mQBNAiqxYTkAAAECALfeHYp0yC80s1ScFvJSpj5eSCAO+hihtneFrrn+vuEcSavh
AAUwpIUGyV2N8n+lFTPnnLc42Ms+c8PJUPYKVI8ABRG0I01hcmMgVGhpYmF1bHQg
PG1hcmNAdGFuZGEuaXNpcy5vcmc+
=HLnv
-----END PGP PUBLIC KEY BLOCK-----""", "PGP_KEY", True),
    ("-----BEGIN PGP SIGNED MESSAGE-----", "PGP_SIGNED_BEGIN", True),
    # WORD
    ("I'm", "WORD", True),
    ("GO", "WORD", True),
    ("wake-up", "WORD", True),
    # NUMBER
    ("555", 'NUM', True),
    ("123523", 'NUM', True),
    ("21.45", 'NUM', True),
    ("23,45000", 'NUM', True),
    (".43", 'NUM', True),
    ("9831231456", 'NUM', False),
    ("52,875,567", 'NUM', True),
    ("52,678,000 89", 'NUM', True),
    ("342 234 545", 'NUM', True),
    ("5k", 'NUM', True),
    ("35G", 'NUM', True),
    ("3/5", 'NUM', True),
    # EMAIL
    ("_1_1_1_@example.com", "EMAIL", True),
    ("email@samlple.ru", "EMAIL", True),
    ("email@sample.com", "EMAIL", True),
    ("email@extra.lol.net", "EMAIL", True),
    ("name-surname@last.ne", "EMAIL", True),
    ("wingo%cspara.decnet@Fedex.Msfc.Nasa.Gov", "EMAIL", True),
    ("willner@head-cfa.harvard.edu", "EMAIL", True),
    # TIME
    ("9:30pm", 'TIME', True),
    ("9:30 pm", 'TIME', True),
    ("18:30", 'TIME', True),
    ("8:30", 'TIME', True),
    ("23:40", 'TIME', True),
    # WEB
    ("https://hello.world", 'WEB', True),
    ("http://lol/lol/lol.com", 'WEB', True),
    ("https://itmo.ru", 'WEB', True),
    # PUNCT
    ("...", "SENT_END", True),
    (",", "PUNC_COMMA", True),
    ("\"", "PUNC_QUOTE", True),
    # EMOT
    (":-)", "EMOT", True),
    # MONTH
    ("Jan", "MONTH", True),
    ("December", "MONTH", True),
    # MONEY
    ("5 USD", "MONEY", True),
    ("5,7 USD", "MONEY", True),
    ("$ 870", "MONEY", True),
    ("$1000", "MONEY", True),
    # PHONE NUM
    ("9831231456", 'PHONE_NUM', True),
    ("706 542-0358", 'PHONE_NUM', True),
    ("(217) 333-6444", 'PHONE_NUM', True),
    ("609-734-6569", 'PHONE_NUM', True),
    # ATTR
    ("Re:\n\n", "ATTR", True),
    ("Line: Test line. \n\n", "ATTR", True),
    ("From: decay@cbnewsj.cb.att.com (dean.kaflowitz). \n\n", "ATTR", True),
])
def test_individual_tokens_are_parsed_correctly(text, tag, exists):
    df = tokenize(text)
    token_found = True if len(df[(df.tag == tag)]) >= 1 else False
    assert token_found and exists or not token_found and not exists, \
        f"token with tag {tag} in text {text} should ex: {exists}:\n{df}"


def test():
    s = """
    HELLO WORLD
    FAKE_END	FAKE_END	FAKE_END	SENT_END
    NEW LINE
    """
    print(s.replace("FAKE_END\tFAKE_END\tFAKE_END\tSENT_END", "\n"))

if __name__ == "__main__":
    unittest.main()
