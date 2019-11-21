from transformers.data.processors.glue import Sst2Processor

class Sst3Processor(Sst2Processor):
    """Processor for SST-3 data set (GLUE version)."""
    def __init__(self):
        super.__init__()
    
    def get_labels(self):
        """See base class."""
        return ["0","1","2"]

