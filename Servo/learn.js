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

var meanDistance = require("../_Utils/validation/meanDistance.js");
var meanPearson = require("../_Utils/validation/meanPearson.js");


var ITER = 100;
var CONTINUOUS_FEATURES = ["pgain", "vgain"];
var CATEGORICAL_FEATURES = ["motor", "screw"];
var TARGET = "class";


var lossWindow = new cnnutil.Window(100);
var lines = 0;
var expected = [];
var predicted = [];
var dataset = [];
var catToVec = {}; // catToVec["screw"]["A"] -> [1, 0, 0]
CATEGORICAL_FEATURES.forEach(function(featureName){
	catToVec[featureName] = {};
});

console.log('READING FILE');

fs.createReadStream("data/servo.csv")
 	.pipe(csv({separator: ",", headers: ["motor", "screw", "pgain", "vgain", "class"]}))
 	.on('data', function(data) {
 		

 		var continuous_features = CONTINUOUS_FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});

		var categorical_features = CATEGORICAL_FEATURES.map(function(feature){
			return data[feature];
		});

		CATEGORICAL_FEATURES.forEach(function(feature){
			if(!catToVec[feature][data[feature]]) {
				// find element 
				var presents = Object.keys(catToVec[feature])
				if (presents.length > 0) {
					// if elements are present push a col
					presents.forEach(function(p){
						catToVec[feature][p].push(0);
					});
					var l = catToVec[feature][presents[0]].length;
					catToVec[feature][data[feature]] = ubique.zeros(1, l - 1)[0].concat(1);
				}
				else
					catToVec[feature][data[feature]] = [1];
			}
		});

		// var x = new convnetjs.Vol(features);
		var y = parseFloat(data[TARGET]);

		dataset.push({continuous_features: continuous_features, categorical_features:categorical_features, target:y});


 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		
		var vectCategoriesSize = Object.keys(catToVec)
			.map(function(feature){
				return Object.keys(catToVec[feature]).length
			})
			.reduce(function(a, b) {
				return a + b;
			});

		var layer_defs = [];
		layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: CONTINUOUS_FEATURES.length + vectCategoriesSize});
		layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
		layer_defs.push({type:'regression', num_neurons: 1});

		var net = new convnetjs.Net();
		net.makeLayers(layer_defs);


		var dataset2 = dataset.map(function(data){
			var catVec = []
			CATEGORICAL_FEATURES.forEach(function(feature, index){
				catVec = catVec.concat(catToVec[feature][data.categorical_features[index]]);
			})
			var tot = data.continuous_features.concat(catVec);

			return {x: new convnetjs.Vol(tot), y: data.target}
		})

		var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 1});

		console.log('LEARNING');
		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {
			// console.log('ITER', iters+1);

			lodash.shuffle(dataset2).forEach(function(line){


				var stats = trainer.train(line.x, [line.y]);
				lossWindow.add(stats.loss);

				var predictObject = net.forward(line.x).w;
				expected.push([line.y]);
				predicted.push([predictObject[0]]);	

				lines += 1;
				if (lines % 1000 === 0){
					console.log("loss", lossWindow.get_average())
					
					var md = meanDistance(expected, predicted);
					var mp = meanPearson(expected, predicted);
					expected = [];
					predicted = [];
					console.log("meanDistance: ", md);
					console.log("meanPearson: ", mp);
				}
			})

		}

		var end = Date.now();
		console.log('Training time for', ITER, 'iterations', (end - start)/1000, 's');

		console.log("===================================================");
		console.log("FINAL EVALUATION");
		expected = [];
		predicted = [];
		dataset2.forEach(function(line){
			var predictObject = net.forward(line.x).w;
			expected.push([line.y]);
			predicted.push([predictObject[0]]);	
			
		});
		var md = meanDistance(expected, predicted);
		var mp = meanPearson(expected, predicted);
		console.log("meanDistance: ", md);
		console.log("meanPearson: ", mp);



		console.log("Saving model");
		var modelJson = net.toJSON();
		var model = "data/model.json"
		var modelPath = path.join(__dirname, model);
		fs.writeFile(modelPath, JSON.stringify(modelJson), function(err) {
			if (err) console.log(err)
			console.log("Model saved in ", modelPath);
		} );
	});



