import unittest
from clean_data import cluster_urls, cluster_users

class TestCleanData(unittest.TestCase):

    def test_cluster_urls(self):
        text = """the\tEng\n…\tO\nhttps\tEng\n//\tO\nt\tEng\n.\tO\nco\tEng\n\nmeta\t48\tpositive"""
        instream = text.split('\n')
        tokens = cluster_urls((line for line in instream))
        assert next(tokens) == "the"
        assert next(tokens) == "…"
        assert next(tokens) == "https//t.co"
        assert next(tokens) == "**DONE**"
        assert next(tokens) == "meta"

    def test_cluster_users(self):
        text = """meta\t48\tpositive\n@\tO\nAbhishar\tHin\n_\tO\nSharma\tHin\n@\tO\nRavishKumarBlog\tHin\nLokSabha\tEng\n\n"""
        instream = text.split('\n')
        tokens = cluster_users((line for line in instream))        
        assert next(tokens) == "meta"
        assert next(tokens) == "positive"
        assert next(tokens) == "@Abhishar_Sharma@RavishKumarBlog"
        assert next(tokens) == "LokSabha"

        text = """meta\t48\tpositive\n@\tO\nAbhisharSharma\tHin\n@\tO\nRavishKumarBlog\tHin\nLokSabha\tEng\n\n"""
        instream = text.split('\n')
        tokens = cluster_users((line for line in instream))        
        assert next(tokens) == "meta"
        assert next(tokens) == "positive"
        assert next(tokens) == "@AbhisharSharma@RavishKumarBlog"
        assert next(tokens) == "LokSabha"

        text = """meta\t48\tpositive\n@\tO\nAbhishar\tHin\n_\tO\nSharma\tHin\nLokSabha\tEng\nme\tHin\njanta\tHin\nsirf\tHin\n\n"""
        instream = text.split('\n')
        tokens = cluster_users((line for line in instream))        
        assert next(tokens) == "meta"
        assert next(tokens) == "positive"
        assert next(tokens) == "@Abhishar_Sharma"
        assert next(tokens) == "LokSabha"
        assert next(tokens) == "me"
        assert next(tokens) == "janta"
        assert next(tokens) == "sirf"

        text = """meta\t3\tnegative\n@\tO\nAdilNisarButt\tHin\npakistan\tHin\nka\tHin\nghra\tHin\ntauq\tHin\nhe\tHin\npakistan\tEng\n\n"""
        instream = text.split('\n')
        tokens = cluster_users((line for line in instream))        
        assert next(tokens) == "meta"
        assert next(tokens) == "negative"
        assert next(tokens) == "@AdilNisarButt"
        assert next(tokens) == "pakistan"
        assert next(tokens) == "ka"
        assert next(tokens) == "ghra"
        assert next(tokens) == "tauq"
        assert next(tokens) == "he"
        assert next(tokens) == "pakistan"


        
if __name__ == "__main__":
	unittest.main()