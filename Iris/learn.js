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
var CONTINUOUS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
var targets = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 3};
var TARGET = "type";

var lines = 0;
var dataset = [];


console.log('READING FILE');


fs.createReadStream("data/iris.csv")
 	.pipe(csv({separator: ",", "headers": ["sepal_length", "sepal_width", "petal_length", "petal_width", "type"]}))
 	.on('data', function(data) {
 		
 		var continuous_features = CONTINUOUS_FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});
 		var x = new convnetjs.Vol(continuous_features)
		
		var y = targets[data[TARGET]];

		dataset.push({
			x: x,
			target: y
		});

 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		
		var layer_defs = [];
		layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: CONTINUOUS_FEATURES.length});
		layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
		layer_defs.push({type:'softmax', num_classes: 3});

		var net = new convnetjs.Net();
		net.makeLayers(layer_defs);

		var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 5});
		var trainAccWindow = new cnnutil.Window(100);
		var lossWindow = new cnnutil.Window(100);
		var f2t = cnnutil.f2t;

		console.log('LEARNING');
		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {

			lodash.shuffle(dataset).forEach(function(line){

				var stats = trainer.train(line.x, [line.target]);
				console.log(line.x);
				lossWindow.add(stats.cost_loss);

				var yhat = net.getPrediction();
				var train_acc = yhat === line.target ? 1.0 : 0.0;
				trainAccWindow.add(train_acc);
				

				lines += 1;
				if (lines % 100 === 0){
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
			

			net.forward(line.x);
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



