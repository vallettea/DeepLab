'use strict';

var PCA = require('ml-pca');
var ubique = require('ubique');

module.exports = function(matrix, k){

	// matrix is of dimension p observations by n features
	// we need to transpose it to access directly the feature vectors
	var matrixT = ubique.transpose(matrix);

	// normalisation of data to get variables with 0 mean and 1 std
	var normalizedMatrixT = matrixT.map(function(featureValues){
		var mean = ubique.mean(featureValues);
		var std = ubique.std(featureValues);

		return featureValues.map(function(value){
			return (value - mean) / std;
		});
	});

	var normalizedMatrix = ubique.transpose(normalizedMatrixT);

	var pca = new PCA(normalizedMatrix);

	var projection = pca.project(normalizedMatrix, k);

	var eigenvectors = pca.getEigenVectors();
	var explainedVariance = pca.getExplainedVariance();

	console.log('eigenvectors', eigenvectors);
	console.log('explainedVariance', explainedVariance);

};

// normalize values

// return k eigenvectors