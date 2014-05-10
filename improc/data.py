from settings import MONGODB_GETTER_SERVER, MONGODB_GETTER_PORT, MONGODB_GETTER_DB, MONGODB_GETTER_COLLECTION

def DataExtractor(object):
  server = None
  port = None
  db = None
  collection = None
  query = {}
  def connect(self):
    connection = pymongo.Connection(self.server, self.port)
    db = connection[self.db
    self.collection = db[self.collection]

  def __init__(self, C = 1, gamma = 0.5):
    pass

def GetterExtractor(DataExtractor):
  query_header = {}
  
  def __init__(self, server = MONGODB_GETTER_SERVER, port = MONGODB_GETTER_PORT, db = MONGODB_GETTER_DB, collection = MONGODB_GETTER_COLLECTION):
    self.server = server
    self.port = port
    self.db = db
    self.collection = collection
    self.xconnect()

  def build(self):
    query = {}

    if len(query_header.keys())
      query["header"] = query_header
