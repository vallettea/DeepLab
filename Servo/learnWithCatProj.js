'use strict';

require("es6-shim");

var fs =  require("fs");
var path = require('path');
var glob = require("glob");
var csv = require('csv-parser');
var convnetjs = require("convnetjs");
var lodash = require("lodash");
var ubique = require("ubique");

var meanDistance = require("../_Utils/validation/meanDistance.js");
var meanPearson = require("../_Utils/validation/meanPearson.js");

var randomProjection = require("../_Utils/preparation/randomProjection.js");


var ITER = 100;
var FEATURES = ["pgain", "vgain", "motor", "screw"];
var CONTINUOUS_FEATURES = ["pgain", "vgain"];
var CATEGORICAL_FEATURES = ["motor", "screw"];
var TARGET = "class";
var PROJ_DIM = 4;


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


var lossWindow = new Window();
var lines = 0;
var expected = [];
var predicted = [];
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

fs.createReadStream("data/servo.csv")
 	.pipe(csv({separator: ",", headers: ["motor", "screw", "pgain", "vgain", "class"]}))
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
		var y = parseFloat(data[TARGET]);

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
		layer_defs.push({type:'regression', num_neurons: 1});

		var net = new convnetjs.Net();
		net.makeLayers(layer_defs);

		var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 1});

		console.log('LEARNING');
		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {
			// console.log('ITER', iters+1);

			lodash.shuffle(dataset).forEach(function(line){

				var features = prepareVolume(line);
				var categorical = line.categorical; // needed because we need to access the projections to update them

				// make a train run on current line
				var stats = trainer.train(features, [line.target]);
				lossWindow.add(stats.loss);

				var predictObject = net.forward(features).w;

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


				expected.push([line.target]);
				predicted.push([predictObject[0]]);

				lines += 1;
				if (lines % 1000 === 0){
					console.log("loss", lossWindow.get_average());
					
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
		dataset.forEach(function(line){
			var features = prepareVolume(line);

			var predictObject = net.forward(features).w;
			expected.push([line.target]);
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



