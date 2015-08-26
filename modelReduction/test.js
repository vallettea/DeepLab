'use strict';

var fs = require('fs');
var csv = require('csv-parser');
var getReducedModel = require('./getReducedModel.js');

var featureMatrix = [];
var FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"];


fs.createReadStream("../data/whites.csv")
 	.pipe(csv({separator: ';'}))
 	.on('data', function(data) {
 		featureMatrix.push(FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		}));
 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		var reducedModel = getReducedModel(featureMatrix, 7);

		console.log("REDUCTION COMPLETE");
		console.log('model length', reducedModel.eigenvectors.length);
		console.log('variance explained', reducedModel.explainedVariance);

	});
