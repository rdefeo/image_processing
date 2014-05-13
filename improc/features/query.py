def do(sample, possibles, comparator):
  """This function does something.

  Args:
     sample (numpy.ndarray):  The item to use as the query
     possibles (dict<numpy.ndarray>):  The item to use as the query
     comparator (DistanceComparator): The comparator to use

  Kwargs:
     state (bool): Current state to be in.

  Returns:


  Raises:

  """
  results = []
  for x in possibles.keys():
    distance = comparator.compare(sample["value"], possibles[x]["value"])
    res = {
      "value": distance,
      "key": x
    }
    results.append(res)

  sorted_results = sorted(results, key=lambda x: x["value"])

  return {
    "comparator": comparator.name,
    "results": sorted_results
  }
