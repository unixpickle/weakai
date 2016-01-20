(function() {

  function BoardView() {
    this._pieces = [];
    this._element = document.getElementById('board');
    for (var i = 0; i < 3; ++i) {
      for (var j = 0; j < 8; ++j) {
        if ((j & 1) !== (i & 1)) {
          continue;
        }
        for (var player = 1; player < 3; ++player) {
          var x = (player === 1 ? j : 7-j);
          var y = (player === 1 ? 7-i : i);
          var piece = new Piece(player, {x: x, y: y});
          this._pieces.push(piece);
          this._element.appendChild(piece.element());
        }
      }
    }
  }

  function Piece(player, position) {
    this._element = document.createElement('div');
    this._element.className = 'board-piece board-piece-player' + player;
    this._player = player;
    this._isKing = false;
    this._position = position;
    this._repositionElement();
  }

  Piece.prototype.element = function() {
    return this._element;
  };

  Piece.prototype.getPlayer = function() {
    return this._player;
  };

  Piece.prototype.getPosition = function() {
    return this._position;
  };

  Piece.prototype.setPosition = function(pos) {
    this._position = pos;
    this._repositionElement();
  };

  Piece.prototype.isKing = function() {
    return this._isKing;
  };

  Piece.prototype.becomeKing = function() {
    if (!this._isKing) {
      this._isKing = true;
      this._element.className += ' board-piece-player' + this._player + '-king';
    }
  };

  Piece.prototype._repositionElement = function() {
    this._element.style.left = ((this._position.x / 8)*100).toFixed(2) + '%';
    this._element.style.top = ((this._position.y / 8)*100).toFixed(2) + '%';
  };

  window.app.BoardView = BoardView;

})();
