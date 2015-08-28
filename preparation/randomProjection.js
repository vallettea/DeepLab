'use strict';

var ubique = require('ubique');

module.exports = function(dataAttributes, dimension){

	var projections = dataAttributes.map(function(attribute){
		
		var projection = ubique.rand([1, dimension])[0]; // ubique.rand() only output matrices ...
		
		// Centering
		projection = ubique.plus(projection, -ubique.mean(projection));
		// Scaling
		projection = ubique.times(projection, 1/ubique.std(projection));

		return projection;
	});

	console.log('Results of random projection', projections);

	return projections;

};