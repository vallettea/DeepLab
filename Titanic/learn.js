'use strict';

require("es6-shim");

var fs =  require("fs");
var path = require('path');
var glob = require("glob");
var csv = require('csv-parser');
var convnetjs = require("convnetjs");
var cnnutil = require("convnetjs/build/util");
var lodash = require("lodash");
var ubique = require("ubique");


var randomProjection = require("../_Utils/preparation/randomProjection.js");


var ITER = 10;
var CONTINUOUS_FEATURES = ["age"];
var CATEGORICAL_FEATURES = ["pclass","embarked","home.dest","room","sex"];
var TARGET = "survived";
var PROJ_DIM = 4;


var lines = 0;
var dataset = [];
var catMap = new Map(); // Map (categoricalFeatureName -> Map(featureAttribute -> []))
CATEGORICAL_FEATURES.forEach(function(featureName){
	catMap.set(featureName, new Map());
});

console.log('READING FILE');

function prepareVolume(line){
	var continuous = line.continuous; // list
	var categorical = line.categorical; // object
	var target = line.target;

	var features = [continuous];

	Object.keys(categorical).forEach(function(featureName){
		features.push(categorical[featureName].projection);
	});
	return new convnetjs.Vol(lodash.flattenDeep(features)); // features is now of dimension inputSize, here -> 100
}

fs.createReadStream("data/titanic.csv")
 	.pipe(csv({separator: ","}))
 	.on('data', function(data) {
 		
 		var continuous_features = CONTINUOUS_FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});

		var categorical_features = {};

		catMap.forEach(function(toProjMap, featureName){

			var attribute = data[featureName];

			if (!toProjMap.has(attribute)) {
				
				// project attribute
				var projection = randomProjection.onItem(PROJ_DIM);
				// update Maps
				toProjMap.set(attribute, projection);
			}
			
			categorical_features[featureName] = {
				attribute: attribute,
				projection: toProjMap.get(attribute)
			};
			
		});
		var y = parseInt(data[TARGET]);

		dataset.push({
			continuous: continuous_features, // list
			categorical: categorical_features, // object
			target: y
		});

 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		
		var inputSize = CONTINUOUS_FEATURES.length + CATEGORICAL_FEATURES.length * PROJ_DIM;

		var layer_defs = [];
		layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: inputSize});
		layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
		layer_defs.push({type:'softmax', num_neurons: 2});

		var net = new convnetjs.Net();
		net.makeLayers(layer_defs);

		var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 1});
		var trainAccWindow = new cnnutil.Window(100);
		var lossWindow = new cnnutil.Window(100);
		var f2t = cnnutil.f2t;

		console.log('LEARNING');
		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {
			// console.log('ITER', iters+1);

			lodash.shuffle(dataset).forEach(function(line){

				var features = prepareVolume(line);
				var categorical = line.categorical; // needed because we need to access the projections to update them

				// make a train run on current line
				var stats = trainer.train(features, line.target);
				// console.log(stats);
				lossWindow.add(stats.cost_loss);

				var dW = {};
				
				CATEGORICAL_FEATURES.forEach(function(featureName, index){
					var grad = [];
					var toProjMap = catMap.get(featureName);
					var attribute = categorical[featureName].attribute;

					// output the gradients as vectors, because convNetJS output is a weird monster...
					for (var i = 0; i<PROJ_DIM; i++){
						var I = CONTINUOUS_FEATURES.length + (index * PROJ_DIM) + i;
						grad.push(features.dw[I]);
					}

					// console.log('attribute', featureName, attribute);
					// console.log('before', toProjMap.get(attribute));

					// update the projections of categorical attributes
					var updatedProjection = ubique.plus(toProjMap.get(attribute), grad);
					toProjMap.set(attribute, updatedProjection);

					// console.log('after', toProjMap.get(attribute));

				});

				var yhat = net.getPrediction();
				var predictObject = net.forward(features, false);
				console.log(yhat, line.target, predictObject)
				var train_acc = yhat === line.target ? 1.0 : 0.0;
				trainAccWindow.add(train_acc);
				

				lines += 1;
				if (lines % 1000 === 0){
					console.log('Training accuracy: ' + f2t(trainAccWindow.get_average()));
					console.log("loss", lossWindow.get_average())
				}
			})

		}

		var end = Date.now();
		console.log('Training time for', ITER, 'iterations', (end - start)/1000, 's');

		console.log("===================================================");
		console.log("FINAL EVALUATION");
		var accuracy = new cnnutil.Window(dataset.length);
		
		dataset.forEach(function(line){
			var features = prepareVolume(line);

			net.forward(features);
			var yhat = net.getPrediction();
			console.log(yhat)
			var train_acc = yhat === line.target ? 1.0 : 0.0;
			accuracy.add(train_acc);
			
		});
		console.log('Training accuracy: ' + f2t(accuracy.get_average()))


		console.log("Saving model");
		var modelJson = net.toJSON();
		var model = "data/model.json"
		var modelPath = path.join(__dirname, model);
		fs.writeFile(modelPath, JSON.stringify(modelJson), function(err) {
			if (err) console.log(err)
			console.log("Model saved in ", modelPath);
		} );
	});



