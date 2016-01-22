(function() {

  function Controls() {
    window.EventEmitter.call(this);
    this._trainButton = document.getElementById('train-button');
    this._recognizeButton = document.getElementById('recognize-button');
    this._snapButton = document.getElementById('snap-button');

    this._canRecognize = false;
    this._snapped = false;
    this._training = false;

    this._setEnabledStates([false, false, false]);
    this._registerEvents();
  }

  Controls.prototype = Object.create(window.EventEmitter.prototype);
  Controls.prototype.constructor = Controls;

  Controls.prototype.enable = function() {
    this._updateState();
  };

  Controls.prototype.setCanRecognize = function(flag) {
    this._canRecognize = flag;
    this._updateState();
  };

  Controls.prototype.setSnapped = function(flag) {
    this._snapped = flag;
    this._updateState();
  };

  Controls.prototype.setTraining = function(flag) {
    this._training = flag;
    this._updateState();
  };

  Controls.prototype._updateState = function() {
    if (this._snapped) {
      this._snapButton.innerText = 'Back to Camera';
      if (this._training) {
        this._setEnabledStates([true, false, false]);
        this._trainButton.innerText = 'Cancel Training';
      } else {
        this._setEnabledStates([true, this._canRecognize, true]);
        this._trainButton.innerText = 'Train';
      }
    } else {
      this._setEnabledStates([false, false, true]);
      this._snapButton.innerText = 'Freeze Picture';
    }
  };

  Controls.prototype._setEnabledStates = function(flags) {
    var buttons = [this._trainButton, this._recognizeButton, this._snapButton];
    for (var i = 0; i < 3; ++i) {
      buttons[i].disabled = !flags[i];
    }
  };

  Controls.prototype._registerEvents = function() {
    this._trainButton.addEventListener('click', function() {
      if (this._training) {
        this.setTraining(false);
        this.emit('cancelTrain');
      } else {
        this.setTraining(true);
        this.emit('train');
      }
    }.bind(this));
    this._recognizeButton.addEventListener('click', this.emit.bind(this, 'recognize'));
    this._snapButton.addEventListener('click', function() {
      if (this._snapped) {
        this.setSnapped(false);
        this.emit('unsnap');
      } else {
        this.setSnapped(true);
        this.emit('snap');
      }
    }.bind(this));
  };

  window.app.Controls = Controls;

})();
