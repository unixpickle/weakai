(function() {

  var DRAWING_COUNT = 3;

  function App() {
    this._drawingsElement = document.getElementById('drawings');
    this._drawings = [];
    for (var i = 0; i < DRAWING_COUNT; ++i) {
      var drawing = new window.app.Canvas();
      this._drawings[i] = drawing;
      this._drawingsElement.appendChild(drawing.element());
    }

    this._mainDrawing = new window.app.Canvas();
    this._mainDrawingElement = document.getElementById('main-drawing');
    this._mainDrawingElement.appendChild(this._mainDrawing.element());

    var vectorSize = this._mainDrawing.dimension() * this._mainDrawing.dimension();
    var network = new window.app.HopfieldNet(vectorSize);
    this._controls = new window.app.Controls(network, this._drawings, this._mainDrawing);
  }

  window.addEventListener('load', function() {
    new App();
  });

})();
