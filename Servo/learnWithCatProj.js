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
	catMap.set(featureName, {
		toProj: new Map(),
		fromProj: new Map()
	});
});

console.log('READING FILE');

fs.createReadStream("data/servo.csv")
 	.pipe(csv({separator: ",", headers: ["motor", "screw", "pgain", "vgain", "class"]}))
 	.on('data', function(data) {
 		
 		var continuous_features = CONTINUOUS_FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});

		var categorical_features = [];

		catMap.forEach(function(maps, featureName){
			var toProjMap = maps.toProj;
			var fromProjMap = maps.fromProj;

			var attribute = data[featureName];

			if (!toProjMap.has(attribute)) {
				
				// project attribute
				var projection = randomProjection.onItem(attribute, PROJ_DIM);
				// update Maps
				toProjMap.set(attribute, projection);
				fromProjMap.set(projection, attribute);
			}
			
			categorical_features.push(toProjMap.get(attribute));
			
		});

		var features = lodash.flattenDeep(continuous_features.concat(categorical_features));

		var x = new convnetjs.Vol(features);
		var y = parseFloat(data[TARGET]);

		dataset.push({x: x, y: y});


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
		dataset.forEach(function(line){
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



