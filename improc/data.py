from settings import MONGODB_GETTER_SERVER, MONGODB_GETTER_PORT, MONGODB_GETTER_DB, MONGODB_GETTER_COLLECTION

def DataExtractor(object):
  server = None
  port = None
  db = None
  collection = None
  query_data = {}
  limit = None

  def connect(self):
    connection = pymongo.Connection(self.server, self.port)
    db = connection[self.db
    self.collection = db[self.collection]

  # def __init__(self, C = 1, gamma = 0.5):
  #   pass

  def execute(self):
    self.build()
    docs = None

    docs = collection.find(
      query_data,
      # {
      #   "shoe": 1
      # }
    ).limit(self.limit)

    return docs



def GetterExtractor(DataExtractor):

  def __init__(self, server = MONGODB_GETTER_SERVER, port = MONGODB_GETTER_PORT, db = MONGODB_GETTER_DB, collection = MONGODB_GETTER_COLLECTION):
    self.server = server
    self.port = port
    self.db = db
    self.collection = collection
    self.xconnect()

  def query(self, _id = None, header = None, detail = None):
    query_data = {}

    if not header == None:
      query_data["header"] = header
    if not detail == None:
      query_data["detail"] = detail
    if not _id == None
      query_data["_id"] = _id

    self.execute()
