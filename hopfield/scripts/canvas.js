(function() {

  var PICTURE_SIZE = 5;
  var CANVAS_SIZE = 100;

  function Canvas() {
    this._cells = [];
    for (var i = 0; i < PICTURE_SIZE*PICTURE_SIZE; ++i) {
      this._cells[i] = false;
    }
    this._element = document.createElement('canvas');
    this._element.width = CANVAS_SIZE;
    this._element.height = CANVAS_SIZE;
    this._element.className = 'drawing';
    this._registerMouseEvents();
  }

  Canvas.prototype.element = function() {
    return this._element;
  };

  Canvas.prototype.vector = function() {
    var res = [];
    for (var i = 0, len = this._cells.length; i < len; ++i) {
      if (this._cells[i]) {
        res[i] = 1;
      } else {
        res[i] = -1;
      }
    }
    return res;
  };

  Canvas.prototype.setVector = function(v) {
    for (var i = 0, len = this._cells.length; i < len; ++i) {
      if (v[i] < 0) {
        this._cells[i] = false;
      } else {
        this._cells[i] = true;
      }
    }
    this._draw();
  };

  Canvas.prototype.dimension = function() {
    return PICTURE_SIZE;
  };

  Canvas.prototype._draw = function() {
    var ctx = this._element.getContext('2d');
    ctx.clearRect(0, 0, this._element.width, this._element.height);
    var tileSize = CANVAS_SIZE / PICTURE_SIZE;
    for (var x = 0; x < PICTURE_SIZE; ++x) {
      for (var y = 0; y < PICTURE_SIZE; ++y) {
        var rectLeft = x*tileSize;
        var rectTop = y*tileSize;
        if (this._cells[x+y*PICTURE_SIZE]) {
          ctx.fillStyle = 'black';
        } else {
          ctx.fillStyle = 'white';
        }
        ctx.fillRect(rectLeft, rectTop, tileSize, tileSize);
      }
    }
  };

  Canvas.prototype._registerMouseEvents = function() {
    var tileSize = CANVAS_SIZE / PICTURE_SIZE;

    var mouseMoved = false;

    this._element.addEventListener('click', function(e) {
      if (mouseMoved) {
        return;
      }
      var clientRect = this._element.getBoundingClientRect();
      var x = Math.floor((e.clientX - clientRect.left) / tileSize);
      var y = Math.floor((e.clientY - clientRect.top) / tileSize);
      x = Math.max(0, Math.min(PICTURE_SIZE-1, x));
      y = Math.max(0, Math.min(PICTURE_SIZE-1, y));
      var idx = x + y*PICTURE_SIZE;
      this._cells[idx] = !this._cells[idx];
      this._draw();
    }.bind(this));

    this._element.addEventListener('mousedown', function(e) {
      mouseMoved = false;
      var boundMovement = function(e) {
        var clientRect = this._element.getBoundingClientRect();
        var x = Math.floor((e.clientX - clientRect.left) / tileSize);
        var y = Math.floor((e.clientY - clientRect.top) / tileSize);
        x = Math.max(0, Math.min(PICTURE_SIZE-1, x));
        y = Math.max(0, Math.min(PICTURE_SIZE-1, y));
        var idx = x + y*PICTURE_SIZE;
        this._cells[idx] = true;
        this._draw();
        mouseMoved = true;
      }.bind(this);
      window.addEventListener('mousemove', boundMovement);
      var boundMouseUp;
      boundMouseUp = function() {
        window.removeEventListener('mousemove', boundMovement);
        window.removeEventListener('mouseup', boundMouseUp);
      }.bind(this);
      window.addEventListener('mouseup', boundMouseUp);
    }.bind(this));
  };

  window.app.Canvas = Canvas;

})();
