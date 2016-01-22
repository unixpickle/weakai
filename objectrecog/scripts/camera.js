(function() {

  var FRAME_RATE = 500;

  function Camera() {
    window.EventEmitter.call(this);
    this._video = null;
    this._paused = false;
  }

  Camera.prototype = Object.create(window.EventEmitter.prototype);
  Camera.prototype.constructor = Camera;

  Camera.prototype.start = function() {
    getUserMedia(function(err, stream) {
      if (err !== null) {
        this.emit('error', err);
      } else {
        this._startWithStream(stream);
      }
    }.bind(this));
  };

  Camera.prototype.pause = function() {
    this._paused = true;
  };

  Camera.prototype.resume = function() {
    this._paused = false;
  };

  Camera.prototype.outputDimensions = function() {
    var width = this._video.videoWidth;
    var height = this._video.videoHeight;
    var scaleFactor = 192 / height;
    return {
      width: Math.round(width * scaleFactor),
      height: Math.round(height * scaleFactor)
    };
  };

  Camera.prototype._startWithStream = function(stream) {
    this._video = document.createElement('video');
    this._video.src = window.URL.createObjectURL(stream);
    this._video.play();

    // On chrome, onloadedmetadata will never be called, so we
    // use a timeout to start emitting frames anyway.
    var loadTimeout = setInterval(function() {
      this._video.onloadedmetadata = null;
      this._beginEmittingFrames();
    }.bind(this), 1000);

    this._video.onloadedmetadata = function() {
      clearTimeout(loadTimeout);
      this._beginEmittingFrames();
    }.bind(this);
  };

  Camera.prototype._beginEmittingFrames = function() {
    this.emit('start', this.outputDimensions());
    setInterval(function() {
      if (!this._paused) {
        this.emit('frame', this._generateFrameCanvas());
      }
    }.bind(this), FRAME_RATE);
  };

  Camera.prototype._generateFrameCanvas = function() {
    var dims = this.outputDimensions();
    var canvas = document.createElement('canvas');
    canvas.width = dims.width;
    canvas.height = dims.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(this._video, 0, 0, dims.width, dims.height);
    return canvas;
  };

  window.app.Camera = Camera;

  function getUserMedia(cb) {
    var gum = (navigator.getUserMedia || navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia || navigator.msGetUserMedia);
    if (!gum) {
      setTimeout(function() {
        cb('getUserMedia() is not available.', null);
      }, 10);
      return;
    }
    gum.call(navigator, {audio: false, video: true},
      function(stream) {
        cb(null, stream);
      },
      function(err) {
        cb(err, null);
      }
    );
  }

})();
