import unittest

from projects.abazarova.source.lab1.tokenizer import *


class TestTokenizer(unittest.TestCase):
    def test_nodes(self):
        self.assertEqual((["2", ".", "3"], "1", "2"), tokenize('"1","2","3"'))

    def test_on_empty_string_with_spaces(self):
        self.assertEqual(1, len(tokenize('"1","   ","     "')[0]))
        self.assertEqual(".", tokenize('"1","   ","     "')[0][0])

    def test_on_backslashes(self):
        self.assertEqual(["2", ".", "foo", "bar", "baz", "qux", "quux"],
                         tokenize('"1","2","foo\\bar\\baz    qux\quux"')[0])

    def test_on_site(self):
        self.assertEqual(["2",".","http://ad.doubleclick.net/ad/idg.us.ifw.general/ibmpseries"],
                         tokenize('"1","2","http://ad.doubleclick.net/ad/idg.us.ifw.general/ibmpseries"')[0])
        self.assertEqual(["2",".","site.com"],tokenize('"1","2","site.com"')[0])

    def test_on_tag(self):
        self.assertEqual(["2", ".", "&lt;p&gt;", "ADVERTISEMENT", "&lt;/p&gt;", "&lt;p&gt;"],
                         tokenize('"1","2","&lt;p&gt;ADVERTISEMENT&lt;/p&gt;&lt;p&gt;"')[0])
        #
        self.assertEqual(["2", ".", "&lt;", "&lt;TAG&gt;", "&gt;"],
                         tokenize('"1","2","&lt;&lt;TAG&gt;&gt;"')[0])

    def test_on_date(self):
        self.assertEqual(["2", ".", "2022/12/12"], tokenize('"1","2","2022/12/12"')[0])

    def test_on_money(self):
        self.assertEqual(["2", ".", "$33"], tokenize('"1","2","$33"')[0])
        self.assertEqual(["2", ".", "$33,000"], tokenize('"1","2","$33,000"')[0])
        self.assertEqual(["2", ".", "$33.9"], tokenize('"1","2","$33.9"')[0])
        self.assertEqual(["2", ".", "$33k"], tokenize('"1","2","$33k"')[0])
        self.assertEqual(["2", ".", "$33b"], tokenize('"1","2","$33b"')[0])
        self.assertEqual(["2", ".", "$33m"], tokenize('"1","2","$33m"')[0])
        self.assertEqual(["2", ".", "$33K"], tokenize('"1","2","$33K"')[0])
        self.assertEqual(["2", ".", "$33B"], tokenize('"1","2","$33B"')[0])
        self.assertEqual(["2", ".", "$33M"], tokenize('"1","2","$33M"')[0])
        self.assertEqual(["2", ".", "$33 billion"], tokenize('"1","2","$33 billion"')[0])
        self.assertEqual(["2", ".", "$33 Billion"], tokenize('"1","2","$33 Billion"')[0])
        self.assertEqual(["2", ".", "$33 million"], tokenize('"1","2","$33 million"')[0])
        self.assertEqual(["2", ".", "$33 Million"], tokenize('"1","2","$33 Million"')[0])
        self.assertEqual(["2", ".", "$33 hundred"], tokenize('"1","2","$33 hundred"')[0])
        self.assertEqual(["2", ".", "$33 Hundred"], tokenize('"1","2","$33 Hundred"')[0])
        self.assertEqual(["2", ".", "$33 thousand"], tokenize('"1","2","$33 thousand"')[0])
        self.assertEqual(["2", ".", "$33 Thousand"], tokenize('"1","2","$33 Thousand"')[0])
        self.assertEqual(["2", ".", "$33,000.9 billion"], tokenize('"1","2","$33,000.9 billion"')[0])

    def test_on_wordnum(self):
        self.assertEqual(["2", ".", "b4"], tokenize('"1","2","b4"')[0])
        self.assertEqual(["2", ".", "7ya"], tokenize('"1","2","7ya"')[0])
        self.assertEqual(["2", ".", "nu3tious"], tokenize('"1","2","nu3tious"')[0])

    def test_on_numer(self):
        self.assertEqual(["2", ".", "#4"], tokenize('"1","2","#4"')[0])
        self.assertEqual(["2", ".", "\'3"], tokenize('"1","2","\'3"')[0])
        self.assertEqual(["2", ".", "4th"], tokenize('"1","2","4th"')[0])
        self.assertEqual(["2", ".", "2nd"], tokenize('"1","2","2nd"')[0])
        self.assertEqual(["2", ".", "3rd"], tokenize('"1","2","3rd"')[0])
        self.assertEqual(["2", ".", "1st"], tokenize('"1","2","1st"')[0])
        self.assertEqual(["2", ".", "No. 4"], tokenize('"1","2","No. 4"')[0])

    def test_on_reduction(self):
        self.assertEqual(["2", ".", "F.B.I."], tokenize('"1","2","F.B.I."')[0])
        self.assertEqual(["2", ".", "A.B"], tokenize('"1","2","A.B"')[0])
        self.assertEqual(["2", ".", "A.B."], tokenize('"1","2","A.B."')[0])
        self.assertEqual(["2", ".", "Azzz.Bzzz"], tokenize('"1","2","Azzz.Bzzz"')[0])
        self.assertEqual(["2", ".", "aaa.bbbb."], tokenize('"1","2","aaa.bbbb."')[0])
        self.assertEqual(["2", ".", "e.t.c"], tokenize('"1","2","e.t.c"')[0])

    def test_on_num(self):
        self.assertEqual(["2", ".", "34"], tokenize('"1","2","34"')[0])
        self.assertEqual(["2", ".", "34.2"], tokenize('"1","2","34.2"')[0])
        self.assertEqual(["2", ".", "-34"], tokenize('"1","2","-34"')[0])
        self.assertEqual(["2", ".", "3/4"], tokenize('"1","2","3/4"')[0])
        self.assertEqual(["2", ".", "3 1/2"], tokenize('"1","2","3 1/2"')[0])
        self.assertEqual(["2", ".", "34 16/9"], tokenize('"1","2","34 16/9"')[0])

    def test_on_punct(self):
        self.assertEqual(["2", ".", "..."], tokenize('"1","2","..."')[0])
        self.assertEqual(["2", ".", "--"], tokenize('"1","2","--"')[0])
        self.assertEqual(["2", ".", "?!"], tokenize('"1","2","?!"')[0])
        self.assertEqual(["2", ".", "a", ",", "b", "?", "c", "!", "d", ";"], tokenize('"1","2","a,b? c! d;"')[0])


if __name__ == "__main__":
    unittest.main()
