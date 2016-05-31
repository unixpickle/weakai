(function() {

  var INTERVAL = 20;
  var ACTIVATION_THRESHOLD = 0;
  var CONVERGE_NODES = 5;

  function HopfieldNet(nodeCount) {
    this._weights = [];
    this._values = [];
    for (var i = 0; i < nodeCount; ++i) {
      this._weights[i] = [];
      this._values[i] = -1;
      for (var j = 0; j < nodeCount; ++j) {
        this._weights[i][j] = 1;
      }
    }
  }

  HopfieldNet.prototype.train = function(samples) {
    this._resetWeights();
    var nodeCount = this._values.length;
    for (var i = 0; i < nodeCount; ++i) {
      for (var j = 0; j < nodeCount; ++j) {
        for (var k = 0, len = samples.length; k < len; ++k) {
          if (samples[k][i] === samples[k][j]) {
            this._weights[i][j] += 1 / len;
          } else {
            this._weights[i][j] -= 1 / len;
          }
        }
      }
    }
  };

  HopfieldNet.prototype.vector = function() {
    return this._values.slice();
  };

  HopfieldNet.prototype.setVector = function(values) {
    for (var i = 0, len = this._values.length; i < len; ++i) {
      this._values[i] = values[i];
    }
  };

  HopfieldNet.prototype.convergeStep = function() {
    for (var x = 0; x < CONVERGE_NODES; ++x) {
      this._convergeRandomNode();
    }
  };

  HopfieldNet.prototype._convergeRandomNode = function() {
    var nodeIndex = Math.floor(Math.random() * this._values.length);
    var sum = 0;
    for (var j = 0, len = this._values.length; j < len; ++j) {
      if (j === nodeIndex) {
        continue;
      }
      var w = this._weights[nodeIndex][j];
      var v = this._values[j];
      sum += w * v;
    }
    if (sum > ACTIVATION_THRESHOLD) {
      this._values[nodeIndex] = 1;
    } else {
      this._values[nodeIndex] = -1;
    }
  };

  HopfieldNet.prototype._resetWeights = function() {
    var nodeCount = this._values.length;
    for (var i = 0; i < nodeCount; ++i) {
      for (var j = 0; j < nodeCount; ++j) {
        this._weights[i][j] = 0;
      }
    }
  };

  window.app.HopfieldNet = HopfieldNet;

})();
