(function() {

  function Cropper(canvas, image) {
    window.EventEmitter.call(this);

    this._cancelled = false;

    this._canvas = canvas;
    this._image = image;

    this._startPoint = null;
    this._currentPoint = null;

    this._mouseDownListener = null;
    this._registerMouseEvents();
    this._draw();
  }

  Cropper.prototype = Object.create(EventEmitter.prototype);
  Cropper.prototype.constructor = Cropper;

  Cropper.prototype.cancel = function() {
    this._cancelled = true;
    this._canvas.removeEventListener('mousedown', this._mouseDownListener);
  };

  Cropper.prototype._draw = function() {
    var ctx = this._canvas.getContext('2d');
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    ctx.drawImage(this._image, 0, 0);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    if (this._startPoint === null) {
      ctx.fillRect(0, 0, this._canvas.width, this._canvas.height);
    } else {
      var rect = this._croppedRect();
      ctx.beginPath();
      ctx.rect(0, 0, this._canvas.width, this._canvas.height);
      ctx.rect(rect.x, rect.y, rect.width, rect.height);
      ctx.fill('evenodd');
    }
  };

  Cropper.prototype._registerMouseEvents = function() {
    this._mouseDownListener = function(e) {
      this._startPoint = this._pointForEvent(e);
      this._currentPoint = this._startPoint;

      var upListener, moveListener;
      moveListener = this._mouseMoveHandler.bind(this);
      upListener = function(e) {
        window.removeEventListener('mouseup', upListener);
        window.removeEventListener('mousemove', moveListener);
        this._mouseUpHandler();
      }.bind(this);

      window.addEventListener('mouseup', upListener);
      window.addEventListener('mousemove', moveListener);
    }.bind(this);
    this._canvas.addEventListener('mousedown', this._mouseDownListener);
  };

  Cropper.prototype._mouseMoveHandler = function(e) {
    if (this._cancelled) {
      return;
    }
    this._currentPoint = this._pointForEvent(e);
    this._draw();
  };

  Cropper.prototype._mouseUpHandler = function() {
    if (this._cancelled) {
      return;
    }
    this.cancel();

    var rect = this._croppedRect();
    if (rect.width === 0 || rect.height === 0) {
      this.emit('cancel');
    } else {
      var image = this._generateCroppedImage(rect);
      this.emit('crop', image);
    }
  };

  Cropper.prototype._generateCroppedImage = function(rect) {
    var resCanvas = document.createElement('canvas');
    resCanvas.width = rect.width;
    resCanvas.height = rect.height;
    resCanvas.getContext('2d').drawImage(this._image, -rect.x, -rect.y);
    return resCanvas;
  };

  Cropper.prototype._croppedRect = function() {
    var x = Math.min(this._currentPoint.x, this._startPoint.x);
    var y = Math.min(this._currentPoint.y, this._startPoint.y);
    var maxX = Math.max(this._currentPoint.x, this._startPoint.x);
    var maxY = Math.max(this._currentPoint.y, this._startPoint.y);

    x = Math.max(0, Math.min(this._canvas.width, x));
    y = Math.max(0, Math.min(this._canvas.width, y));
    maxX = Math.max(0, Math.min(this._canvas.height, maxX));
    maxY = Math.max(0, Math.min(this._canvas.height, maxY));

    return {x: x, y: y, width: Math.max(0, maxX-x), height: Math.max(0, maxY-y)};
  };

  Cropper.prototype._pointForEvent = function(e) {
    var b = this._canvas.getBoundingClientRect();
    return {x: e.clientX - b.left, y: e.clientY - b.top};
  };

  window.app.Cropper = Cropper;

})();
