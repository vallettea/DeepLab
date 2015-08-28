'use strict';

var ubique = require('ubique');

/*

This function projects categorical attributes into a multidimensional space.
Each attribute gets a vector generated using random distribution centered and scaled.
This is to be used with the MLP method on categorical inputs as explained in http://arxiv.org/pdf/1508.00021v1.pdf

*/

module.exports = function(dataAttributes, dimension){

	var projections = dataAttributes.map(function(attribute){
		
		var projection = ubique.rand(1, dimension)[0]; // ubique.rand() only output matrices ...
		
		// Centering
		projection = ubique.plus(projection, -ubique.mean(projection));
		// Scaling
		projection = ubique.times(projection, 1/ubique.std(projection));

		return projection;
	});

	console.log('Results of random projection', projections);

	return projections;

}