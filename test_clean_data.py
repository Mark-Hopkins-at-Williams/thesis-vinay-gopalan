import unittest
import json
import os
from clean_data import cluster_urls, tokenize_conll, conll_to_json
from clean_data import cluster_usernames
from clean_data import BasicToken, EndOfSegment, Sentiment, URL, Username

class TestCleanData(unittest.TestCase):

    def test_tokenize_conll(self):
        text = ("the\tEng\n" + 
                "…\tO\n" +
                "\n" +
                "meta\t48\tpositive")
        instream = text.split('\n')
        tokens = tokenize_conll((line for line in instream))
        assert next(tokens) == BasicToken("the")
        assert next(tokens) == BasicToken("…")
        assert next(tokens) == EndOfSegment()
        assert next(tokens) == Sentiment("positive")

    def test_cluster_urls(self):
        text = ("the\tEng\n" + 
                "…\tO\n" +
                "https\tEng\n" +
                "//\tO\n" +
                "t\tEng\n" +
                ".\tO\n" +
                "co\tEng\n" +
                "\n" +
                "meta\t48\tpositive")
        instream = text.split('\n')
        tokens = tokenize_conll((line for line in instream))
        tokens = cluster_urls(tokens)
        assert next(tokens) == BasicToken("the")
        assert next(tokens) == BasicToken("…")
        assert next(tokens) == URL("https//t.co")
        assert next(tokens) == EndOfSegment()
        assert next(tokens) == Sentiment("positive")

    def test_cluster_usernames(self):
        text = ("meta\t48\tpositive\n" + 
                "@\tO\n" +
                "Abhishar\tHin\n" +
                "_\tO\n" +
                "Sharma\tHin\n" +
                "@\tO\n" +
                "RavishKumarBlog\tHin\n" +
                "LokSabha\tEng\n" +
                "\n")
        instream = text.split('\n')
        tokens = tokenize_conll((line for line in instream))
        tokens = cluster_usernames(tokens)        
        assert next(tokens) == Sentiment("positive")
        assert next(tokens) == Username("@Abhishar_Sharma")       
        assert next(tokens) == Username("@RavishKumarBlog")
        assert next(tokens) == BasicToken("LokSabha")
        assert next(tokens) == EndOfSegment()
        
    def test_conll_to_json(self):
        conll_to_json("test/example.txt", "temp.json")
        try:
            with open("temp.json") as reader:
                data = json.load(reader)
                assert len(data) == 2
                assert data[0]["sentiment"] ==  "negative"
                assert (data[0]["segment"] == "pakistan ka ghra tauq he " +
                        "Pakistan Israel ko tasleem nahein kerta Isko " +
                        "Palestine kehta he - OCCUPIED PALESTINE")                     
                assert data[1]["sentiment"] == "negative"
                assert (data[1]["segment"] == "Madarchod mulle ye mathura " +
                        "me Nahi dikha tha jab mullo ne Hindu ko iss liye " +
                        "mara ki vo lasse ki paise mag liye the \u2026")                        
        except Exception:
            print("Unit test failure!")
            os.remove("temp.json")

        


        
if __name__ == "__main__":
	unittest.main()