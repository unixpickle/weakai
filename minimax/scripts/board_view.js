(function() {

  function BoardView() {
    this._pieces = {};
    this._element = document.getElementById('board');

    var gameState = new window.app.GameState();
    for (var x = 0; x < window.app.GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < window.app.GameState.BOARD_SIZE; ++y) {
        var state = gameState.pieceAtPosition({x: x, y: y});
        if (state === null) {
          continue;
        }
        var piece = new Piece(state.getPlayer(), {x: x, y: y});
        this._element.appendChild(piece.element());
        this._pieces[state.getId()] = piece;
      }
    }
  }

  BoardView.prototype.updateWithState = function(gameState) {
    this._element.innerHTML = '';
    for (var x = 0; x < window.app.GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < window.app.GameState.BOARD_SIZE; ++y) {
        var state = gameState.pieceAtPosition({x: x, y: y});
        if (state === null) {
          continue;
        }
        var piece = this._pieces[state.getId()];
        piece.setPosition({x: x, y: y});
        if (state.isKing() && !piece.isKing()) {
          piece.becomeKing();
        }
        this._element.appendChild(piece.element());
      }
    }
  };

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
