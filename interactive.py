"""
File: interactive.py
Name: 許宏裕
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

import submission
import util


def main():
	numEpochs = 40
	alpha = 0.01

	#1.Reading data
	trainExamples = util.readExamples('polarity.train')
	validationExamples = util.readExamples('polarity.dev')

	#2. Selecting feature extractor
	featureExtractor = submission.extractWordFeatures

	#3. Training model and getting weights
	weights = submission.learnPredictor(
		trainExamples,
		validationExamples,
		featureExtractor,
		numEpochs,
		alpha
	)

	#4.Entering into interactive mode
	util.interactivePrompt(featureExtractor, weights)

if __name__ == '__main__':
	main()
