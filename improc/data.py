import pymongo

from settings import MONGODB_GETTER_SERVER, MONGODB_GETTER_PORT, MONGODB_GETTER_DB, MONGODB_GETTER_COLLECTION

class DataExtractor(object):
  server = None
  port = None
  db = None
  collection_name = None
  query_data = {}
  limit = None

  def connect(self):
    connection = pymongo.Connection(self.server, self.port)
    db = connection[self.db]
    self.collection = db[self.collection_name]

  # def __init__(self, C = 1, gamma = 0.5):
  #   pass

  def execute(self):
    print self.query_data
    query = self.collection.find(
      self.query_data,
      # {
      #   "shoe": 1
      # }
    )
    if self.limit:
      query = query.limit(self.limit)

    docs = []

    for x in query:
      docs.append(x)

    return docs



class GetterExtractor(DataExtractor):

  def __init__(self, server = MONGODB_GETTER_SERVER, port = MONGODB_GETTER_PORT, db = MONGODB_GETTER_DB, collection = MONGODB_GETTER_COLLECTION):
    self.server = server
    self.port = port
    self.db = db
    self.collection_name = collection
    self.connect()

  def query(self, _id = None, header = None, detail = None, limit = None):
    self.limit = limit
    if not header == None:
      # self.query_data["header"] = {}
      for x in header.keys():
        self.query_data["header.%s" % x] = header[x]
    if not detail == None:
      self.query_data["detail"] = detail
    if not _id == None:
      self.query_data["_id"] = _id

    return self.execute()
