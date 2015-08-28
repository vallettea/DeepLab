'use strict';

require("es6-shim");

var fs =  require("fs");
var path = require('path');
var glob = require("glob");
var csv = require('csv-parser');
var convnetjs = require("convnetjs");
var lodash = require("lodash");

var meanDistance = require("../../_Utils/validation/meanDistance.js");
var meanPearson = require("../../_Utils/validation/meanPearson.js");


var ITER = 20;
var L1_DEPTH = 4; 
var OUT_DEPTH = 1; 

var FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"];
var TARGET = "quality";

var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 1});


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
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: FEATURES.length});
layer_defs.push({type:'fc', num_neurons: L1_DEPTH, activation:'sigmoid'});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:10, activation:'relu'});
// layer_defs.push({type:'fc', num_neurons:10, activation:'relu'});
layer_defs.push({type:'regression', num_neurons: OUT_DEPTH});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// var trainer = new convnetjs.Trainer(net, {method: 'adagrad', l2_decay: 0.001, l1_decay: 0.001, batch_size: 1});
var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, l1_decay: 0.001, batch_size: 1});

var lossWindow = new Window();
var lines = 0;
var expected = [];
var predicted = [];
var dataset = [];

console.log('READING FILE');


fs.createReadStream("../data/whites.csv")
 	.pipe(csv({separator: ';'}))
 	.on('data', function(data) {

 		var features = FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});
		var x = new convnetjs.Vol(features);
		var y = parseFloat(data[TARGET]);

		dataset.push({x:x, y:y});


 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		console.log('LEARNING');
		var start = Date.now();

		for(var iters=0; iters<ITER; iters++) {
			console.log('ITER', iters+1);

			lodash.shuffle(dataset).forEach(function(line){
				var stats = trainer.train(line.x, [line.y]);
				lossWindow.add(stats.loss);

				var predictObject = net.forward(line.x).w;
				expected.push([line.y]);

				predicted.push([predictObject[0]]);

				lines += 1;
				if (lines % 1000 === 0){
					var md = meanDistance(expected, predicted);
					var mp = meanPearson(expected, predicted);
					expected = [];
					predicted = [];

					// console.log('gradients', line.x.dw);

					// console.log(lines, "lines --> meanDistance: ", md, "meanPearson: ", mp, "loss", lossWindow.get_average());
					
				}
			})

		}

		var end = Date.now();
		console.log('Training time for', ITER, 'iterations', (end - start)/1000, 's');

		console.log("===================================================");
		console.log("FINAL EVALUATION:");

		start = Date.now();

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

		var end = Date.now();
		console.log('Evaluating time', (end - start)/1000, 's');

		console.log("SAVING MODEL");
		var modelJson = net.toJSON();
		var model = "../data/model.json"
		var modelPath = path.join(__dirname, model);
		fs.writeFile(modelPath, JSON.stringify(modelJson), function(err) {
			if (err) console.log(err)
			console.log("Model saved in ", modelPath);
		} );
	});



