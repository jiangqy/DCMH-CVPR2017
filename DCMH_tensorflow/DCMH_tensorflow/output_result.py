import pickle
import pprint


fp = open(‘./log/result_13-Apr-04-1492073645_16bits_MIRFLICKR-25K.pkl', 'rb')

result = pickle.load(fp)

pprint.pprint(result)
