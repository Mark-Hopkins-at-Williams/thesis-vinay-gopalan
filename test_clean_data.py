import unittest
from clean_data import cluster_urls

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
        
  

if __name__ == "__main__":
	unittest.main()
