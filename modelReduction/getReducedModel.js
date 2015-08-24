'use strict';

var PCA = require('ml-pca');
var ubique = require('ubique');

module.exports = function(matrix, k){

	// normalization is done directly by ml-pca
	var pca = new PCA(matrix);

	var eigenvectors = pca.getEigenvectors();
	var reducedEigenvectors = ubique.linspace(0, k - 1, k).map(function(i){
		return eigenvectors[i];
	});

	var reducedExplainedVariance = ubique.sum(pca.getExplainedVariance().slice(0, k));

	// console.log('reduced eigenvectors', reducedEigenvectors);
	// console.log('reduced explainedVariance', reducedExplainedVariance);

	return {
		eigenvectors: reducedEigenvectors,
		explainedVariance: reducedExplainedVariance,
		projection: pca.project(matrix, k)
	};
};
