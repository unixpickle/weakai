(function() {

  var INTERVAL = 100;

  function Controls(network, trainingSamples, drawing) {
    this._network = network;
    this._trainingSamples = trainingSamples;
    this._drawing = drawing;

    this._timer = null;
    this._toggleButton = document.getElementById('toggle-clock');
    this._trainButton = document.getElementById('train-network');
    this._registerButtons();
  }

  Controls.prototype._timerTick = function() {
    this._network.setVector(this._drawing.vector());
    this._network.convergeStep();
    this._drawing.setVector(this._network.vector());
  };

  Controls.prototype._train = function() {
    var samples = [];
    for (var i = 0, len = this._trainingSamples.length; i < len; ++i) {
      samples[i] = this._trainingSamples[i].vector();
    }
    this._network.train(samples);
  };

  Controls.prototype._registerButtons = function() {
    this._toggleButton.addEventListener('click', function() {
      if (this._timer === null) {
        this._toggleButton.innerText = 'Stop Running';
        this._timer = setInterval(this._timerTick.bind(this), INTERVAL);
      } else {
        this._toggleButton.innerText = 'Start Running';
        clearInterval(this._timer);
        this._timer = null;
      }
    }.bind(this));
    this._trainButton.addEventListener('click', this._train.bind(this));
  };

  window.app.Controls = Controls;

})();
