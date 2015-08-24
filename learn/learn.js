'use strict';

require("es6-shim");

var fs =  require("fs");
var path = require('path');
var glob = require("glob");
var csv = require('csv-parser');
var convnetjs = require("convnetjs");
var lodash = require("lodash");

var meanDistance = require("../validation/meanDistance.js");
var meanPearson = require("../validation/meanPearson.js");

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
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: FEATURES.length});
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
// layer_defs.push({type:'fc', num_neurons:20, activation:'sigmoid', drop_prob: 0.5});
// layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
layer_defs.push({type:'regression', num_neurons: 1});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

var trainer = new convnetjs.Trainer(net, {method: 'adelta', l2_decay: 0.001, batch_size: 1});

var lossWindow = new Window();
var lines = 0;
var expected = [];
var predicted = [];


fs.createReadStream("../data/whites.csv")
 	.pipe(csv({separator: ';'}))
 	.on('data', function(data) {

 		var features = FEATURES.map(function(feature){
			return parseFloat(data[feature]);
		});
		var x = new convnetjs.Vol(features);
		var y = parseFloat(data.quality);

		var stats = trainer.train(x, y);
		lossWindow.add(stats.loss);

		var predictObject = net.forward(x).w;
		expected.push([y]);
		predicted.push([predictObject["0"]]);	
		console.log(predictObject["0"])

		lines += 1;
		if (lines % 100 === 0){
			console.log("loss", lossWindow.get_average())
			
			var md = meanDistance(expected, predicted);
			var mp = meanPearson(expected, predicted);
			expected = [];
			predicted = [];
			console.log("meanDistance: ", md);
			console.log("meanPearson: ", mp);
		}


 	})
 	.on('error', function(err){
  		console.log(err.message);
	})
	.on("end", function(){
		console.log("Saving model");
		var modelJson = net.toJSON();
		var model = "../data/model.json"
		var modelPath = path.join(__dirname, model);
		fs.writeFile(modelPath, JSON.stringify(modelJson), function(err) {
			if (err) console.log(err)
			console.log("Model saved in ", modelPath);
		} );
	});



