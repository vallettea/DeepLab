'use strict';

require("es6-shim");

var fs =  require("fs");
var path = require('path');
var glob = require("glob");
var csv = require('csv-parser');
var convnetjs = require("convnetjs");
var lodash = require("lodash");

var getReducedModel = require('../modelReduction/getReducedModel.js');
var meanDistance = require("../validation/meanDistance.js");
var meanPearson = require("../validation/meanPearson.js");

var MODEL_SIZE = 11; // 11 is full size
var ITER = 20;
var FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"];

// error window
var Window = function(size, minsize) {
    this.v = [];
    this.size = typeof(size)==='undefined' ? 100 : size;
    this.minsize = typeof(minsize)==='undefined' ? 10 : minsize;
    this.sum = 0;
  }

Window.prototype = {
    add: function(x) {
      this.v.push(x);
      this.sum += x;
      if(this.v.length>this.size) {
        var xold = this.v.shift();
        this.sum -= xold;
      }
    },
    get_average: function() {
      if(this.v.length < this.minsize) return -1;
      else return this.sum/this.v.length;
    },
    reset: function(x) {
      this.v = [];
      this.sum = 0;
    }
}


var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: MODEL_SIZE});
layer_defs.push({type:'fc', num_neurons: 10, activation:'relu'});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'relu'});
// layer_defs.push({type:'fc', num_neurons:10, activation:'relu'});
layer_defs.push({type:'regression', num_neurons: 1});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// var trainer = new convnetjs.Trainer(net, {method: 'adagrad', l2_decay: 0.001, l1_decay: 0.001, batch_size: 1});
var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, l1_decay: 0.001, batch_size: 1});

var lossWindow = new Window();
var lines = 0;
var featureMatrix = [];
var obsVector = [];

var expectedVector = []; // same as obsVector, but with [] for each item
var predictedVector = [];
var dataset = [];

console.log('READING FILE');

fs.createReadStream("../data/whites.csv")
 	.pipe(csv({separator: ';'}))
 	.on('data', function(data) {
 		//  Creation of feature Matrix and observations vector
 		featureMatrix.push(FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		}));

		obsVector.push(parseFloat(data['quality']));

 	// 	var features = FEATURES.map(function(feature){
		// 	return parseFloat(data[feature]);
		// });
		// var x = new convnetjs.Vol(features);
		// var y = parseFloat(data.quality);

		// dataset.push({x:x, y:y});


 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){

		// Model reduction
		var reduction = getReducedModel(featureMatrix, MODEL_SIZE);
		var reducedMatrix = reduction.projection;

		console.log("REDUCTION COMPLETE");
		console.log('model length', reduction.eigenvectors.length);
		console.log('variance explained', reduction.explainedVariance);

		// format to dataSet
		var dataset = reducedMatrix.to2DArray().map(function(features, index){
			return {
				x: new convnetjs.Vol(features),
				y: obsVector[index]
			};
		});

		// Learning
		console.log('LEARNING');

		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {
			console.log('ITER', iters+1);

			lodash.shuffle(dataset).forEach(function(line){
				var stats = trainer.train(line.x, [line.y]);

				lossWindow.add(stats.loss);

				var predictObject = net.forward(line.x).w;
				expectedVector.push([line.y]);
				predictedVector.push([predictObject[0]]);

				lines ++;

				if (lines % 1000 === 0){
					var md = meanDistance(expectedVector, predictedVector);
					var mp = meanPearson(expectedVector, predictedVector);
					expectedVector = [];
					predictedVector = [];
					// console.log(lines, "lines --> meanDistance: ", md, "meanPearson: ", mp, "loss", lossWindow.get_average());
					
				}
			});
		}

		var end = Date.now();
		console.log('Training time for', ITER, 'iterations', (end - start)/1000, 's');

		console.log("===================================================");
		console.log("FINAL EVALUATION");
		expectedVector = [];
		predictedVector = [];

		start = Date.now();

		dataset.forEach(function(line){
			var predictObject = net.forward(line.x).w;
			expectedVector.push([line.y]);
			predictedVector.push([predictObject[0]]);	
			
		});
		var md = meanDistance(expectedVector, predictedVector);
		var mp = meanPearson(expectedVector, predictedVector);
		console.log("meanDistance: ", md);
		console.log("meanPearson: ", mp);

		end = Date.now();
		console.log('Evaluation time', (end - start)/1000, 's');


		console.log("SAVING MODEL");
		var modelJson = net.toJSON();
		var model = "../data/modelWithReduction.json"
		var modelPath = path.join(__dirname, model);
		fs.writeFile(modelPath, JSON.stringify(modelJson), function(err) {
			if (err) console.log(err)
			console.log("Model saved in ", modelPath);
		} );
	});



