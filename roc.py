import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def cal_metric(groundTruth, predicted):
	fpr, tpr, thresholds = roc_curve(groundTruth, predicted)
	y = (tpr)
	x = (fpr)
	z = tpr +fpr
	tpr = tpr.reshape((tpr.shape[0],1))
	fpr = fpr.reshape((fpr.shape[0],1))
	xnew = np.arange(0, 1, 0.0000001)
	func = interpolate.interp1d(x, y)
	# frr = fpr
	ynew = func(xnew)
	
	znew = abs(xnew + ynew-1)

	eer=xnew[np.argmin(znew)]
	print(eer)
	# interpolate thresholds
	func_2 = interpolate.interp1d(x, thresholds)
	thresholds_new = func_2(xnew)

	print("Threshold at eer: {}".format(thresholds_new[np.argmin(znew)]))

	FPR = {"TPR(1.%)": 0.01, "TPR(.5%)": 0.005}

	TPRs = {"TPR(1.%)": 0.01, "TPR(.5%)": 0.005}
	for i, (key, value) in enumerate(FPR.items()):

		index = np.argwhere(xnew == value)

		score = ynew[index] 

		TPRs[key] = float(np.squeeze(score))
#	    print(key, score)
	if 0:
		plt.plot(xnew, ynew)
		plt.title("ROC curve")
		plt.xlabel("FPR")
		plt.ylabel("TPR")
		plt.show()
	
	auc = roc_auc_score(groundTruth, predicted)

	return eer,TPRs, auc, {'x':xnew, 'y':ynew}

groundTruth = [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
predicted = [0.8, 0.1, 0.1, 0.05, 0.3, 0.5, 0.2, 0.3, 0.79, 0.98, 0.7, 0.2, 0.95]
cal_metric(groundTruth, predicted)


# from scipy.optimize import brentq
# from scipy.interpolate import interp1d
# from sklearn.metrics import roc_curve


# def cal_metric(groundTruth, predicted):
# 	fpr, tpr, thresholds = roc_curve(groundTruth, predicted, pos_label=1)

# 	eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
# 	thresh = interp1d(fpr, thresholds)(eer)

# 	print(eer)
# 	print(thresh)

# groundTruth = [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
# predicted = [0.8, 0.1, 0.1, 0.05, 0.3, 0.5, 0.2, 0.3, 0.79, 0.98, 0.7, 0.2, 0.95]
# cal_metric(groundTruth, predicted)