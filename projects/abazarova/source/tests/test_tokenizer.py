import unittest

from projects.abazarova.source.lab1.tokenizer import *


class TestTokenizer(unittest.TestCase):
    def test_nodes(self):
        self.assertEqual((["2", ".", "\n", "3"], "1", "2"), tokenize('"1","2","3"'))

    def test_on_empty_string_with_spaces(self):
        self.assertEqual(2, len(tokenize('"1","   ","     "')[0]))
        self.assertEqual([".", "\n"], tokenize('"1","   ","     "')[0])

    def test_on_backslashes(self):
        self.assertEqual(["2", ".", "\n", "foo", "bar", "baz", "qux", "quux"],
                         tokenize('"1","2","foo\\bar\\baz    qux\quux"')[0])

    def test_on_site(self):
        self.assertEqual(["2", ".", "\n", "http://ad.doubleclick.net/ad/idg.us.ifw.general/ibmpseries"],
                         tokenize('"1","2","http://ad.doubleclick.net/ad/idg.us.ifw.general/ibmpseries"')[0])
        self.assertEqual(["2", ".", "\n", "https://site.com"], tokenize('"1","2","https://site.com"')[0])
        self.assertEqual(["2", ".", "\n", "site.com"], tokenize('"1","2","site.com"')[0])

    def test_on_tag(self):
        self.assertEqual(["2", ".", "\n", "&lt;p&gt;", "ADVERTISEMENT", "&lt;/p&gt;", "&lt;p&gt;"],
                         tokenize('"1","2","&lt;p&gt;ADVERTISEMENT&lt;/p&gt;&lt;p&gt;"')[0])
        self.assertEqual(["2", ".", "\n", "&lt;", "&lt;TAG&gt;", "&gt;"],
                         tokenize('"1","2","&lt;&lt;TAG&gt;&gt;"')[0])

    def test_on_date(self):
        self.assertEqual(["2", ".", "\n", "2022/12/12"], tokenize('"1","2","2022/12/12"')[0])

    def test_on_money(self):
        self.assertEqual(["2", ".", "\n", "$33"], tokenize('"1","2","$33"')[0])
        self.assertEqual(["2", ".", "\n", "$33,000"], tokenize('"1","2","$33,000"')[0])
        self.assertEqual(["2", ".", "\n", "$33.9"], tokenize('"1","2","$33.9"')[0])
        self.assertEqual(["2", ".", "\n", "$33k"], tokenize('"1","2","$33k"')[0])
        self.assertEqual(["2", ".", "\n", "$33b"], tokenize('"1","2","$33b"')[0])
        self.assertEqual(["2", ".", "\n", "$33m"], tokenize('"1","2","$33m"')[0])
        self.assertEqual(["2", ".", "\n", "$33K"], tokenize('"1","2","$33K"')[0])
        self.assertEqual(["2", ".", "\n", "$33B"], tokenize('"1","2","$33B"')[0])
        self.assertEqual(["2", ".", "\n", "$33M"], tokenize('"1","2","$33M"')[0])
        self.assertEqual(["2", ".", "\n", "$33 billion"], tokenize('"1","2","$33 billion"')[0])
        self.assertEqual(["2", ".", "\n", "$33 Billion"], tokenize('"1","2","$33 Billion"')[0])
        self.assertEqual(["2", ".", "\n", "$33 million"], tokenize('"1","2","$33 million"')[0])
        self.assertEqual(["2", ".", "\n", "$33 Million"], tokenize('"1","2","$33 Million"')[0])
        self.assertEqual(["2", ".", "\n", "$33 hundred"], tokenize('"1","2","$33 hundred"')[0])
        self.assertEqual(["2", ".", "\n", "$33 Hundred"], tokenize('"1","2","$33 Hundred"')[0])
        self.assertEqual(["2", ".", "\n", "$33 thousand"], tokenize('"1","2","$33 thousand"')[0])
        self.assertEqual(["2", ".", "\n", "$33 Thousand"], tokenize('"1","2","$33 Thousand"')[0])
        self.assertEqual(["2", ".", "\n", "$33,000.9 billion"], tokenize('"1","2","$33,000.9 billion"')[0])

    def test_on_wordnum(self):
        self.assertEqual(["2", ".", "\n", "b4"], tokenize('"1","2","b4"')[0])
        self.assertEqual(["2", ".", "\n", "7ya"], tokenize('"1","2","7ya"')[0])
        self.assertEqual(["2", ".", "\n", "nu3tious"], tokenize('"1","2","nu3tious"')[0])

    def test_on_numer(self):
        self.assertEqual(["2", ".", "\n", "#4"], tokenize('"1","2","#4"')[0])
        self.assertEqual(["2", ".", "\n", "\'3"], tokenize('"1","2","\'3"')[0])
        self.assertEqual(["2", ".", "\n", "4th"], tokenize('"1","2","4th"')[0])
        self.assertEqual(["2", ".", "\n", "2nd"], tokenize('"1","2","2nd"')[0])
        self.assertEqual(["2", ".", "\n", "3rd"], tokenize('"1","2","3rd"')[0])
        self.assertEqual(["2", ".", "\n", "1st"], tokenize('"1","2","1st"')[0])
        self.assertEqual(["2", ".", "\n", "No. 4"], tokenize('"1","2","No. 4"')[0])

    def test_on_reduction(self):
        self.assertEqual(["2", ".", "\n", "F.B.I."], tokenize('"1","2","F.B.I."')[0])
        self.assertEqual(["2", ".", "\n", "A.B"], tokenize('"1","2","A.B"')[0])
        self.assertEqual(["2", ".", "\n", "A.B."], tokenize('"1","2","A.B."')[0])
        self.assertEqual(["2", ".", "\n", "Azzz.Bzzz"], tokenize('"1","2","Azzz.Bzzz"')[0])
        self.assertEqual(["2", ".", "\n", "aaa.bbbb."], tokenize('"1","2","aaa.bbbb."')[0])
        self.assertEqual(["2", ".", "\n", "e.t.c"], tokenize('"1","2","e.t.c"')[0])

    def test_on_num(self):
        self.assertEqual(["2", ".", "\n", "34"], tokenize('"1","2","34"')[0])
        self.assertEqual(["2", ".", "\n", "34.2"], tokenize('"1","2","34.2"')[0])
        self.assertEqual(["2", ".", "\n", "-34"], tokenize('"1","2","-34"')[0])
        self.assertEqual(["2", ".", "\n", "3/4"], tokenize('"1","2","3/4"')[0])
        self.assertEqual(["2", ".", "\n", "3 1/2"], tokenize('"1","2","3 1/2"')[0])
        self.assertEqual(["2", ".", "\n", "34 16/9"], tokenize('"1","2","34 16/9"')[0])

    def test_on_punct(self):
        self.assertEqual(["2", ".", "\n", "...", "\n"], tokenize('"1","2","..."')[0])
        self.assertEqual(["2", ".", "\n", "--"], tokenize('"1","2","--"')[0])
        self.assertEqual(["2", ".", "\n", "?!", "\n"], tokenize('"1","2","?!"')[0])
        self.assertEqual(["2", ".", "\n", "a", ",", "b", "?", "\n", "c", "!", "\n", "d", ";"], tokenize('"1","2","a,b? c! d;"')[0])

    def test_sep(self):
        self.assertEqual(["2", ".", "\n", "word", "...", "\n", "otter", "!?", "\n", "rob"],
                         tokenize('"1","2","word... otter!? rob"')[0])

    def test_stem(self):
        self.assertEqual("go", stemm("going"))
        self.assertEqual("student", stemm("students"))
        self.assertEqual("123", stemm("123"))

    def test_pos(self):
        self.assertEqual([("I", "PRP"), ("am", "VBP"), ("going", "VBG"), ("to", "TO"), ("school", "NN")],
                         pos(["I", "am", "going", "to", "school"]))

    def test_lemm(self):
        self.assertEqual("go", lemm("going", "VBG"))
        self.assertEqual("be", lemm("am", "VBP"))
        self.assertEqual("student", lemm("students", "NN"))

    def sep_tok(self):
        self.assertEqual(["a","b","c"], sent_tok("a.b?c!"))


if __name__ == "__main__":
    unittest.main()
