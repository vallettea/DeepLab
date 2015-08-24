'use strict';

var ubique = require('ubique');
var stats = require('simple-statistics');

/*

The input is a list of pairs (Xi, Yi) where Xi is the vector of observation i and Yi is the vector of corresponding prediction i.
We have p observations => 0 <= i < p

Each Xi or Yi is composed of n features which represent the coordinates in the reduced vector space, each of them called xij or yij

*/


module.exports = function(observations, predictions){

	var transposedObs = ubique.transpose(observations);
	var transposedPreds = ubique.transpose(predictions);

	var pearsons = [];

	transposedPreds.forEach(function(predVector, j){ // j is from 0 to n-1

		var obsVector = transposedObs[j];

		var oStdDev = stats.sampleStandardDeviation(obsVector);
		var pStdDev = stats.sampleStandardDeviation(predVector);

		var opCovariance = stats.sampleCovariance(obsVector, predVector);

		var pearson = opCovariance / (oStdDev * pStdDev);
		pearsons.push(pearson);

	});

	return ubique.mean(pearsons);

};