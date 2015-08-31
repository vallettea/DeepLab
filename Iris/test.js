'use strict';

require("es6-shim");

var convnetjs = require("convnetjs");

var net = new convnetjs.Net();
var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});
 

  var layer_defs = [];
  layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:4});
  layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
  layer_defs.push({type:'softmax', num_classes:3});
  net.makeLayers(layer_defs);

  
  // tanh are their own layers. Softmax gets its own fully connected layer.
  // this should all get desugared just fine.
  console.log(net.layers.length, 7); 
  

  var x = new convnetjs.Vol([ 5.8, 2.7, 4.1, 1 ]);
  var probability_volume = net.forward(x);

  console.log(probability_volume.w); 
