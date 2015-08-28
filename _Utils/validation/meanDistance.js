'use strict';

var ubique = require('ubique');

/*

The input is a list of pairs (Xi, Yi) where Xi is the vector of observation i and Yi is the vector of corresponding prediction i.
We have p observations => 0 <= i < p

Each Xi or Yi is composed of n features which represent the coordinates in the reduced vector space, each of them called xij or yij

*/


module.exports = function(observations, predictions){

	var distances = [];

	predictions.forEach(function(predVector, index){

		var obsVector = observations[index];

		distances.push(ubique.pdist(obsVector, predVector, 'euclidean'));

	});

	return ubique.mean(distances);
}
