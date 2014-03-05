import cv2
import numpy as np
# class LetterStatModel(object):
#     class_n = 26
#     train_ratio = 0.5
# 
#     def load(self, fn):
#         self.model.load(fn)
#     def save(self, fn):
#         self.model.save(fn)
# 
#     def unroll_samples(self, samples):
#         sample_n, var_n = samples.shape
#         new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
#         new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
#         new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
#         return new_samples
# 
#     def unroll_responses(self, responses):
#         sample_n = len(responses)
#         new_responses = np.zeros(sample_n*self.class_n, np.int32)
#         resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
#         new_responses[resp_idx] = 1
#         return new_responses
class StatModel(object):
  class_n = 26
  train_ratio = 0.5
  def load(self, fn):
      self.model.load(fn)
  def save(self, fn):
      self.model.save(fn)
  def unroll_samples(self, samples):
      sample_n, var_n = samples.shape
      new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
      new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
      new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
      return new_samples

  def unroll_responses(self, responses):
      sample_n = len(responses)
      new_responses = np.zeros(sample_n*self.class_n, np.int32)
      resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
      new_responses[resp_idx] = 1
      return new_responses

class Boost(StatModel):
    def __init__(self):
        self.model = cv2.Boost()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred

class RTrees(StatModel):
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        # self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

# class KNearest(LetterStatModel):
#     def __init__(self):
#         self.model = cv2.KNearest()
# 
#     def train(self, samples, responses):
#         self.model.train(samples, responses)
# 
#     def predict(self, samples):
#         retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
#         return results.ravel()

        
class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        # self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)
        # self.model.train_auto(samples, responses)
    def predict(self, samples):
        return self.model.predict_all(samples).ravel()
    def predictSingle(self, sample):
        return int(self.model.predict(sample, True))