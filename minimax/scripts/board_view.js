(function() {

  function BoardView() {
    window.EventEmitter.call(this);

    this._pieces = {};
    this._element = document.getElementById('board');

    var gameState = new window.app.GameState();
    for (var x = 0; x < window.app.GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < window.app.GameState.BOARD_SIZE; ++y) {
        var state = gameState.pieceAtPosition({x: x, y: y});
        if (state === null) {
          continue;
        }
        var piece = new Piece(state.getId(), state.getPlayer(), {x: x, y: y});
        this._element.appendChild(piece.element());
        this._pieces[state.getId()] = piece;
      }
    }

    this._registerMouseEvents();
  }

  BoardView.prototype = Object.create(window.EventEmitter.prototype);
  BoardView.prototype.constructor = BoardView;

  BoardView.prototype.element = function() {
    return this._element;
  };

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

  BoardView.prototype._registerMouseEvents = function() {
    this._element.addEventListener('mousedown', this._startMouseDragging.bind(this));
  };

  BoardView.prototype._startMouseDragging = function(e) {
    // This prevents the cursor from being an IBeam
    e.preventDefault();

    var movePiece = null;
    var startPosition = null;
    var pieceIds = Object.keys(this._pieces);
    for (var i = 0, len = pieceIds.length; i < len; ++i) {
      var piece = this._pieces[pieceIds[i]];
      if (!piece.element().parentNode || piece.getPlayer() !== 1) {
        continue;
      }
      var pieceRect = piece.element().getBoundingClientRect();
      var pieceSize = piece.element().offsetWidth;
      if (e.clientX >= pieceRect.left && e.clientY >= pieceRect.top &&
          e.clientX < pieceRect.left+pieceSize &&
          e.clientY < pieceRect.top+pieceSize) {
        movePiece = piece;
        startPosition = piece.getPosition();
        break;
      }
    }

    if (movePiece === null) {
      return;
    }

    var upHandler, moveHandler;
    upHandler = function() {
      window.removeEventListener('mouseup', upHandler);
      window.removeEventListener('mousemove', moveHandler);
      this._finishMovingPiece(movePiece);
    }.bind(this);
    moveHandler = function(newEvent) {
      var width = this._element.offsetWidth;
      var height = this._element.offsetHeight;
      var xOffset = newEvent.clientX - e.clientX;
      var yOffset = newEvent.clientY - e.clientY;
      var scaler = window.app.GameState.BOARD_SIZE / width;
      movePiece.setPosition({
        x: startPosition.x + xOffset*scaler,
        y: startPosition.y + yOffset*scaler
      });
    }.bind(this);
    window.addEventListener('mouseup', upHandler);
    window.addEventListener('mousemove', moveHandler);
  };

  BoardView.prototype._finishMovingPiece = function(piece) {
    var position = piece.getPosition();
    var roundedPos = {
      x: Math.round(position.x),
      y: Math.round(position.y)
    };
    this.emit('move', piece.getId(), roundedPos);
  };

  function Piece(id, player, position) {
    this._id = id;
    this._element = document.createElement('div');
    this._element.className = 'board-piece board-piece-player' + player;
    this._element.style.pointerEvents = 'none';
    this._player = player;
    this._isKing = false;
    this._position = position;
    this._repositionElement();
  }

  Piece.prototype.element = function() {
    return this._element;
  };

  Piece.prototype.getId = function() {
    return this._id;
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
    var size = window.app.GameState.BOARD_SIZE;
    this._element.style.left = ((this._position.x/size)*100).toFixed(2) + '%';
    this._element.style.top = ((this._position.y/size)*100).toFixed(2) + '%';
  };

  window.app.BoardView = BoardView;

})();
