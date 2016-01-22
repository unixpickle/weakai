(function() {

  var camera = null;
  var controls = null;
  var canvas = null;
  var cropper = null;
  var lastFrame = null;

  function App() {
    this._camera = new window.app.Camera();
    this._controls = new window.app.Controls();
    this._canvas = document.getElementById('canvas');

    this._cropper = null;
    this._lastFrame = null;
    this._trainedImage = null;

    this._registerCameraEvents();
    this._registerControlEvents();

    this._camera.start();
  }

  App.prototype.beginTraining = function() {
    this._cropper = new window.app.Cropper(this._canvas, this._lastFrame);

    this._cropper.on('cancel', function() {
      this._cropper = null;
      this._controls.setTraining(false);
      this._drawLastFrame();
    }.bind(this));

    this._cropper.on('crop', function(image) {
      this._cropper = null;
      this._trainedImage = image;
      this._controls.setTraining(false);
      this._controls.setCanRecognize(true);
      this._drawLastFrame();
    }.bind(this));
  };

  App.prototype.recognize = function() {
    var match = window.app.findSubImage(this._canvas, this._trainedImage);

    this._drawLastFrame();
    var ctx = this._canvas.getContext('2d');
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.rect(match.x, match.y, this._trainedImage.width, this._trainedImage.height);
    ctx.stroke();
  };

  App.prototype._registerCameraEvents = function() {
    this._camera.on('error', function() {
      document.body.innerHTML = '<div id="error">Camera API not supported</div>';
    });

    this._camera.on('start', function(dimensions) {
      this._canvas.width = dimensions.width;
      this._canvas.height = dimensions.height;
      this._canvas.style.left = 'calc(50% - ' + Math.round(dimensions.width/2) + 'px)';
    }.bind(this));

    this._camera.on('frame', function(frame) {
      this._controls.enable();
      this._lastFrame = frame;
      this._drawLastFrame();
    }.bind(this));
  };

  App.prototype._registerControlEvents = function() {
    this._controls.on('snap', this._camera.pause.bind(this._camera));
    this._controls.on('unsnap', this._camera.resume.bind(this._camera));
    this._controls.on('train', this.beginTraining.bind(this));
    this._controls.on('cancelTrain', function() {
      this._cropper.cancel();
      this._cropper = null;
      this._drawLastFrame();
    }.bind(this));
    this._controls.on('recognize', this.recognize.bind(this));
  };

  App.prototype._drawLastFrame = function() {
    this._canvas.getContext('2d').drawImage(this._lastFrame, 0, 0);
  };

  window.addEventListener('load', function() {
    new App();
  });

})();
