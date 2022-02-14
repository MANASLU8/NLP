import pandas as pd
from pytest import mark

from task1.tokenizer import tokenize_text
from task1.token_tag import TokenTag


@mark.parametrize("raw_text,expected_token_tag,should_exist", [
    # --- NUMBER ---
    # simple
    ("123", TokenTag.NUMBER, True),
    ("123456789012345678901234567890", TokenTag.NUMBER, True),
    # floats
    ("123.4", TokenTag.NUMBER, True),
    ("123.4567890123456789", TokenTag.NUMBER, True),
    (".4", TokenTag.NUMBER, True),
    (".456789", TokenTag.NUMBER, True),
    ("123 456.4", TokenTag.NUMBER, True),
    ("123,456,789.4", TokenTag.NUMBER, True),
    # digit grouping
    ("123 456", TokenTag.NUMBER, True),
    ("123 456 789", TokenTag.NUMBER, True),
    ("123.456.789", TokenTag.NUMBER, True),
    ("123,456,789", TokenTag.NUMBER, True),
    ("3,456,789", TokenTag.NUMBER, True),
    ("23 456 789", TokenTag.NUMBER, True),
    ("23 456 789 271 381 371 271 236 281 261", TokenTag.NUMBER, True),
    # fractions
    ("5 3/4", TokenTag.NUMBER, True),
    ("135 000 3/4", TokenTag.NUMBER, True),
    ("5/3", TokenTag.NUMBER, True),
    ("5 3//4", TokenTag.NUMBER, False),
    ("5 3/", TokenTag.NUMBER, False),
    ("5/3/7", TokenTag.NUMBER, False),
    # coefficients
    ("1k", TokenTag.NUMBER, True),
    ("1m", TokenTag.NUMBER, True),
    ("1M", TokenTag.NUMBER, True),
    ("1G", TokenTag.NUMBER, True),
    ("1.3m", TokenTag.NUMBER, True),
    ("500k", TokenTag.NUMBER, True),
    ("1230m", TokenTag.NUMBER, True),
    # counts
    ("1st", TokenTag.NUMBER, True),
    ("2nd", TokenTag.NUMBER, True),
    ("3rd", TokenTag.NUMBER, True),
    ("12th", TokenTag.NUMBER, True),
    ("12132nd", TokenTag.NUMBER, True),
    ("12 132nd", TokenTag.NUMBER, True),
    # --- MONEY ---
    ("5$", TokenTag.MONEY, True),
    ("5 $", TokenTag.MONEY, True),
    ("5 USD", TokenTag.MONEY, True),
    ("5 CAD", TokenTag.MONEY, True),
    ("$5", TokenTag.MONEY, True),
    ("$ 5", TokenTag.MONEY, True),
    ("$5.32", TokenTag.MONEY, True),
    ("$5.000", TokenTag.MONEY, True),
    ("5k$", TokenTag.MONEY, True),
    ("$1.2m", TokenTag.MONEY, True),
    ("USD 5,000,000", TokenTag.MONEY, True),
    ("$.02", TokenTag.MONEY, True),
    ("US$5", TokenTag.MONEY, True),
    ("US$5", TokenTag.MONEY, True),
    ("US$          40.00", TokenTag.MONEY, True),
    ("$", TokenTag.MONEY, False),
    ("$$", TokenTag.MONEY, False),
    ("$", TokenTag.MONEY, False),
    # --- PHONE NUMBER ---
    ("(602)-554-2685", TokenTag.PHONE_NUMBER, True),
    ("(619)455-8600", TokenTag.PHONE_NUMBER, True),
    ("(33) 76 39 78 92", TokenTag.PHONE_NUMBER, True),
    ("(705) 748-1653", TokenTag.PHONE_NUMBER, True),
    ("408-262-1469", TokenTag.PHONE_NUMBER, True),
    ("+44 81 528 9864", TokenTag.PHONE_NUMBER, True),
    ("+61-8-259-6486", TokenTag.PHONE_NUMBER, True),
    ("+49 521-106-5375", TokenTag.PHONE_NUMBER, True),
    ("+43 (1) 89100", TokenTag.PHONE_NUMBER, True),
    ("+43 (1) 89100 / 3961", TokenTag.PHONE_NUMBER, True),
    ("(+47) 67125580 ext. 211", TokenTag.PHONE_NUMBER, True),
    ("(617) 890 1100 ext.7531", TokenTag.PHONE_NUMBER, True),
    ("(617)8901100/7531", TokenTag.PHONE_NUMBER, True),
    # --- TIME ---
    ("9am", TokenTag.TIME, True),
    ("9 am", TokenTag.TIME, True),
    ("9 AM", TokenTag.TIME, True),
    ("9pm", TokenTag.TIME, True),
    ("9:30pm", TokenTag.TIME, True),
    ("9:30 pm", TokenTag.TIME, True),
    ("18:30", TokenTag.TIME, True),
    ("18:3", TokenTag.TIME, False),
    ("8:30", TokenTag.TIME, True),
    ("8:30:32", TokenTag.TIME, True),
    ("18:30:32", TokenTag.TIME, True),
    ("24:40", TokenTag.TIME, False),
    ("23:40", TokenTag.TIME, True),
    ("32:40", TokenTag.TIME, False),
    ("23:60", TokenTag.TIME, False),
    ("23:59:60", TokenTag.TIME, False),
    # --- QUOTE HEADER ---
    ("In article <1993Apr20.010326.8634@csus.edu>, arthurc@sfsuvax1.sfsu.edu (Arthur Chandler) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In <1993Apr20.001428.724@indyvax.iupui.edu> tffreeba@indyvax.iupui.edu writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <1qve4kINNpas@sal-sun121.usc.edu> schaefer@sal-sun121.usc.edu (Peter Schaefer) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <C5LJG5.17n.1@cs.cmu.edu> mwm+@cs.cmu.edu (Mark Maimone) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <schumach.734984753@convex.convex.com> schumach@convex.com (Richard A. Schumacher) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <ugo62B8w165w@angus.mi.org> dragon@angus.mi.org writes",
     TokenTag.QUOTE_HEADER, True),
    ("ajjb@adam4.bnsc.rl.ac.uk (Andrew Broderick) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <1993Apr15.051746.29848@news.duc.auburn.edu>, snydefj@eng.auburn.edu writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <1993Apr20.152819.28186@ke4zv.uucp> gary@ke4zv.UUCP (Gary Coffman) writes",
     TokenTag.QUOTE_HEADER, True),
    ("fils@iastate.edu (Douglas R Fils) writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <19930423.010821.639@almaden.ibm.com> nicho@vnet.ibm.com writes",
     TokenTag.QUOTE_HEADER, True),
    ("In article <C5qqxp.IE1@cbmvax.cbm.commodore.com>, h@cbmvax.cbm.commodore.com (Jerry Hartzler - CATS) writes",
     TokenTag.QUOTE_HEADER, True),
    ("hartzler@cbmvax.cbm.commodore.com (Jerry Hartzler - CATS CATS CATS) writes",
     TokenTag.QUOTE_HEADER, True),
    # --- URL/EMAIL ---
    # urls - https://mathiasbynens.be/demo/url-regex
    ("foo.com", TokenTag.URL_EMAIL, True),
    ("http://foo.com", TokenTag.URL_EMAIL, True),
    ("http://foo.com/blah_blah", TokenTag.URL_EMAIL, True),
    ("http://foo.com/blah_blah/", TokenTag.URL_EMAIL, True),
    ("http://www.example.com/wpstyle/?p=364", TokenTag.URL_EMAIL, True),
    ("https://www.example.com/foo/?bar=baz&inga=42&quux", TokenTag.URL_EMAIL, True),
    ("http://userid:password@example.com:8080/", TokenTag.URL_EMAIL, True),
    ("https://142.42.1.1:8080/", TokenTag.URL_EMAIL, True),
    ("ftp://foo.bar/baz", TokenTag.URL_EMAIL, True),
    # emails - https://gist.github.com/cjaoude/fd9910626629b53c4d25
    ("email@example.com", TokenTag.URL_EMAIL, True),
    ("firstname.lastname@example.com", TokenTag.URL_EMAIL, True),
    ("email@subdomain.example.com", TokenTag.URL_EMAIL, True),
    ("firstname+lastname@example.com", TokenTag.URL_EMAIL, True),
    ("email@123.123.123.123", TokenTag.URL_EMAIL, True),
    ("1234567890@example.com", TokenTag.URL_EMAIL, True),
    ("email@example-one.com", TokenTag.URL_EMAIL, True),
    ("_______@example.com", TokenTag.URL_EMAIL, True),
    ("email@example.name", TokenTag.URL_EMAIL, True),
    ("email@example.museum", TokenTag.URL_EMAIL, True),
    ("email@example.co.jp", TokenTag.URL_EMAIL, True),
    ("firstname-lastname@example.com", TokenTag.URL_EMAIL, True),
    # selected emails from dataset
    ("1993Apr26.193924.1189@bnr.ca", TokenTag.URL_EMAIL, True),
    ("1993Apr27.094238.7682@samba.oit.unc.edu", TokenTag.URL_EMAIL, True),
    ("Bruce.Scott@launchpad.unc.edu", TokenTag.URL_EMAIL, True),
    ("atae@spva.ph.ic.ac.uk", TokenTag.URL_EMAIL, True),
    ("ralph.buttigieg@f635.n713.z3.fido.zeta.org.au", TokenTag.URL_EMAIL, True),
    ("0004244402@mcimail.com", TokenTag.URL_EMAIL, True),
    ("wingo%cspara.decnet@Fedex.Msfc.Nasa.Gov", TokenTag.URL_EMAIL, True),
    ("29APR199321594919@kelvin.jpl.nasa.gov", TokenTag.URL_EMAIL, True),
    ("1rnaih$jvj@access.digex.net", TokenTag.URL_EMAIL, True),
    ("willner@head-cfa.harvard.edu", TokenTag.URL_EMAIL, True),
    # --- PUNCT ---
    (".", TokenTag.PUNCT_SENTENCE, True),
    ("?", TokenTag.PUNCT_SENTENCE, True),
    ("!", TokenTag.PUNCT_SENTENCE, True),
    ("?!", TokenTag.PUNCT_SENTENCE, True),
    ("!?", TokenTag.PUNCT_SENTENCE, True),
    ("?!?!?!?!", TokenTag.PUNCT_SENTENCE, True),
    ("...", TokenTag.PUNCT_SENTENCE, True),
    ("???", TokenTag.PUNCT_SENTENCE, True),
    ("!!!", TokenTag.PUNCT_SENTENCE, True),
    ("................", TokenTag.PUNCT_SENTENCE, False),
    ("\"", TokenTag.PUNCT_QUOTES, True),
    ("«", TokenTag.PUNCT_QUOTES, True),
    ("(", TokenTag.PUNCT_BRACES, True),
    (")", TokenTag.PUNCT_BRACES, True),
    (":", TokenTag.PUNCT_COLON, True),
    # --- EMOTICON ---
    (":3", TokenTag.EMOTICON, True),
    (":)", TokenTag.EMOTICON, True),
    (":0", TokenTag.EMOTICON, True),
    (":D", TokenTag.EMOTICON, True),
    ("xD", TokenTag.EMOTICON, True),
    ("XD", TokenTag.EMOTICON, True),
    (":-)", TokenTag.EMOTICON, True),
    (":(", TokenTag.EMOTICON, True),
    (":'(", TokenTag.EMOTICON, True),
    ("¯\(ツ)/¯", TokenTag.EMOTICON, True),
    # --- PGP ---
    ("-----BEGIN PGP SIGNED MESSAGE-----", TokenTag.PGP_BEGINNING, True),
    ("""-----BEGIN PGP SIGNATURE-----
Version: 2.2

iQBuAgUBK8DNazh0K1zBsGrxAQFoZQLEC/XKXMoDhCPf/AZ3aOQSLfz+6w400UDk
Ng6prxnPuEuSZQEiiusMCVcRcGnWbaVrxFjA1o4yubh01Czcg3ZC9wLJolXlxJn7
iSJh/eTZxmJnNynJxlGs0Ao=
=4eZb
-----END PGP SIGNATURE-----""", TokenTag.PGP_SIGNATURE, True),
    ("""-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: 2.0

mQBNAiqxYTkAAAECALfeHYp0yC80s1ScFvJSpj5eSCAO+hihtneFrrn+vuEcSavh
AAUwpIUGyV2N8n+lFTPnnLc42Ms+c8PJUPYKVI8ABRG0I01hcmMgVGhpYmF1bHQg
PG1hcmNAdGFuZGEuaXNpcy5vcmc+
=HLnv
-----END PGP PUBLIC KEY BLOCK-----""", TokenTag.PGP_PUBLIC_KEY, True),
    # --- WORD ---
    ("test", TokenTag.WORD, True),
    ("testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest", TokenTag.WORD, True),
    ("non-practical", TokenTag.WORD, True),
    ("don't", TokenTag.WORD, True),
    ("TSV", TokenTag.WORD, True),
    ("3D", TokenTag.WORD, True),
])
def test_individual_tokens_are_parsed_correctly(raw_text: str, expected_token_tag: str, should_exist: bool):
    result = tokenize_text(raw_text)
    if len(result) > 0:
        is_token_found = len(result[(result.token == raw_text) & (result.tag == expected_token_tag)]) == 1
    else:
        is_token_found = False
    assert is_token_found and should_exist or not is_token_found and not should_exist, \
        f"{raw_text} was incorrectly tokenized to this (should_exist={should_exist}):\n{result}"


@mark.parametrize("raw_text,expected_token_tag,expected_count", [
    ("What?! There's no way: too quick... Reset it, faster!", TokenTag.PUNCT_SENTENCE, 4),
    ("What?! There's no way: too quick... Reset it, faster!", TokenTag.PUNCT_COLON, 1),
    ("What?! There's no way: too quick... Reset it, faster!", TokenTag.WORD, 9),
    ("It's a 3D thing.", TokenTag.NUMBER, 0),
    ("His birthday is 1987-03-01.", TokenTag.NUMBER, 0),
    ("Don't be \"nice\", that's too artificial.", TokenTag.WORD, 6),
    ("Don't be \"nice\", that's too artificial.", TokenTag.PUNCT_QUOTES, 2),
])
def test_tokens_in_context_are_parsed_correctly(raw_text: str, expected_token_tag: str, expected_count: int):
    result = tokenize_text(raw_text)
    if len(result) > 0:
        found_count = len(result[result.tag == expected_token_tag])
    else:
        found_count = 0
    assert found_count == expected_count, \
        f"{raw_text} was incorrectly tokenized to this (expected_count={expected_count}):\n{result}"


_EXAMPLE_TEXT = """
In <C5LJ0t.K52@blaze.cs.jhu.edu> eifrig@beanworld.cs.jhu.edu (Jonathan Eifrig) writes:
> FACT: that's not 3D!

No, that's actually not true. Visit google.com.
Btw, selling cookies :) $.03 a piece.

---------------------
John Hesse           
jhesse@netcom.com        
Tel.:+49-40-54715-224 ext.4
Fax: +49(1)54715-226
Fachbereich Informatik - AGN
---------------------
A man,     
    a plan, 
        a canal, Bob.
---------------------
"""

_EXAMPLE_TOKENS = pd.DataFrame([
    ("In <C5LJ0t.K52@blaze.cs.jhu.edu> eifrig@beanworld.cs.jhu.edu (Jonathan Eifrig) writes", TokenTag.QUOTE_HEADER),
    (":", TokenTag.PUNCT_COLON),
    ("FACT", TokenTag.WORD),
    (":", TokenTag.PUNCT_COLON),
    ("that's", TokenTag.WORD),
    ("not", TokenTag.WORD),
    ("3D", TokenTag.WORD),
    ("!", TokenTag.PUNCT_SENTENCE),
    ("No", TokenTag.WORD),
    (",", TokenTag.PUNCT_SENTENCE),
    ("that's", TokenTag.WORD),
    ("actually", TokenTag.WORD),
    ("not", TokenTag.WORD),
    ("true", TokenTag.WORD),
    (".", TokenTag.PUNCT_SENTENCE),
    ("Visit", TokenTag.WORD),
    ("google.com", TokenTag.URL_EMAIL),
    (".", TokenTag.PUNCT_SENTENCE),
    ("Btw", TokenTag.WORD),
    (",", TokenTag.PUNCT_SENTENCE),
    ("selling", TokenTag.WORD),
    ("cookies", TokenTag.WORD),
    (":)", TokenTag.EMOTICON),
    ("$.03", TokenTag.MONEY),
    ("a", TokenTag.WORD),
    ("piece", TokenTag.WORD),
    (".", TokenTag.PUNCT_SENTENCE),
    ("John", TokenTag.WORD),
    ("Hesse", TokenTag.WORD),
    ("jhesse@netcom.com", TokenTag.URL_EMAIL),
    ("Tel", TokenTag.WORD),
    (".", TokenTag.PUNCT_SENTENCE),
    (":", TokenTag.PUNCT_COLON),
    ("+49-40-54715-224 ext.4", TokenTag.PHONE_NUMBER),
    ("Fax", TokenTag.WORD),
    (":", TokenTag.PUNCT_COLON),
    ("+49(1)54715-226", TokenTag.PHONE_NUMBER),
    ("Fachbereich", TokenTag.WORD),
    ("Informatik", TokenTag.WORD),
    ("AGN", TokenTag.WORD),
    ("A", TokenTag.WORD),
    ("man", TokenTag.WORD),
    (",", TokenTag.PUNCT_SENTENCE),
    ("a", TokenTag.WORD),
    ("plan", TokenTag.WORD),
    (",", TokenTag.PUNCT_SENTENCE),
    ("a", TokenTag.WORD),
    ("canal", TokenTag.WORD),
    (",", TokenTag.PUNCT_SENTENCE),
    ("Bob", TokenTag.WORD),
    (".", TokenTag.PUNCT_SENTENCE),
], columns=["token", "tag"])


def test_example_text_is_tokenized_correctly():
    result = tokenize_text(_EXAMPLE_TEXT)
    assert result.equals(_EXAMPLE_TOKENS), f"Example text was incorrectly tokenized to this:\n{result}"
