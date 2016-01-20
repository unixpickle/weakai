(function() {

  function GameState() {
    this._playerTurn = 1;
    this._specificJumpingPosition = null;

    this._board = [];
    for (var i = 0; i < GameState.BOARD_SIZE*GameState.BOARD_SIZE; ++i) {
      this._board[i] = null;
    }

    var idStepper = 0;
    for (var i = 0; i < GameState.PLAYER_ROWS; ++i) {
      for (var j = 0; j < GameState.BOARD_SIZE; ++j) {
        if ((j & 1) !== (i & 1)) {
          continue;
        }
        for (var player = 1; player <= 2; ++player) {
          var x = (player === 1 ? j : GameState.BOARD_SIZE-1-j);
          var y = (player === 1 ? GameState.BOARD_SIZE-1-i : i);
          var piece = new PieceState(idStepper++, player, false);
          this._board[x + y*GameState.BOARD_SIZE] = piece;
        }
      }
    }
  }

  GameState.BOARD_SIZE = 8;
  GameState.PLAYER_ROWS = 3;

  GameState.prototype.playerTurn = function() {
    return this._playerTurn;
  };

  GameState.prototype.pieceAtPosition = function(p) {
    return this._board[p.x + p.y*GameState.BOARD_SIZE];
  };

  GameState.prototype.availableMoves = function() {
    if (this._specificJumpingPosition !== null) {
      return this._availableJumpsForPosition(this._specificJumpingPosition);
    }

    var jumps = this._availableJumps();
    if (jumps.length > 0) {
      return jumps;
    }

    return this._availableNonJumps();
  };

  GameState.prototype.stateAfterMove = function(move) {
    var state = Object.create(GameState);
    state._playerTurn = this._playerTurn;
    state._specificJumpingPosition = null;
    state._board = this._board.slice();

    state._setPieceAtPosition(move.getSource(), null);
    var jumpedPos = move.getJumpedPosition();
    if (jumpedPos !== null) {
      state._setPieceAtPosition(jumpedPos, null);
    }

    if ((move.getDestination().y === 0 && state._playerTurn === 1) ||
        (move.getDestination().y === GameState.BOARD_SIZE-1 && state._playerTurn === 2)) {
      var newPiece = new PieceState(move.getPiece().getId(), move.getPiece().getPlayer(), true);
      state._setPieceAtPosition(move.getDestination(), newPiece);
    } else {
      state._setPieceAtPosition(move.getDestination(), move.getPiece());
    }

    if (jumpedPos !== null && state._availableJumpsForPosition(move.getDestination()).length > 0) {
      state._specificJumpingPosition = move.getDestination();
    } else {
      state._playerTurn = (this._playerTurn === 1 ? 2 : 1);
      state._specificJumpingPosition = null;
    }
  };

  GameState.prototype._availableJumps = function() {
    var res = [];
    for (var x = 0; x < GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < GameState.BOARD_SIZE; ++y) {
        var jumps = this._availableJumpsForPosition({x: x, y: y});
        for (var i = 0, len = jumps.length; i < len; ++i) {
          res.push(jumps[i]);
        }
      }
    }
    return res;
  };

  GameState.prototype._availableJumpsForPosition = function(piecePosition) {
    var res = [];
    var piece = this.pieceAtPosition(piecePosition);

    if (piece === null || piece.getPlayer() !== this._playerTurn) {
      return res;
    }

    for (var dx = -1; dx <= 1; dx += 2) {
      if (piecePosition.x+dx*2 < 0 || piecePosition.x+dx*2 > GameState.BOARD_SIZE-1) {
        continue;
      }
      for (var dy = -1; dy <= 1; dy += 2) {
        if ((piecePosition.y+dy*2 < 0 || piecePosition.y+dy*2 > GameState.BOARD_SIZE-1) ||
            (dy === -1 && !piece.isKing() && piece.getPlayer() === 2) ||
            (dy === 1 && !piece.isKing() && piece.getPlayer() === 1)) {
          continue;
        }
        var jumpPiece = this.pieceAtPosition({x: piecePosition.x + dx, y: piecePosition.y + dy});
        if (jumpPiece === null || jumpPiece.getPlayer() === this._playerTurn) {
          continue;
        }
        var destinationPoint = {x: piecePosition.x + dx*2, y: piecePosition.y + dy*2};
        var destination = this.pieceAtPosition(destinationPoint);
        if (destination === null) {
          res.push(new Move(piece, piecePosition, destinationPoint));
        }
      }
    }

    return res;
  };

  GameState.prototype._availableNonJumps = function() {
    var res = [];
    for (var x = 0; x < GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < GameState.BOARD_SIZE; ++y) {
        var moves = this._availableNonJumpsForPosition({x: x, y: y});
        for (var i = 0, len = moves.length; i < len; ++i) {
          res.push(moves[i]);
        }
      }
    }
    return res;
  };

  GameState.prototype._availableNonJumpsForPosition = function(piecePosition) {
    var res = [];
    var piece = this.pieceAtPosition(piecePosition);
    if (piece === null || piece.getPlayer() !== this._playerTurn) {
      return res;
    }
    for (var dx = -1; dx <= 1; dx += 2) {
      if (piecePosition.x+dx < 0 || piecePosition.x+dx > GameState.BOARD_SIZE-1) {
        continue;
      }
      for (var dy = -1; dy <= 1; dy += 2) {
        if ((piecePosition.y+dy < 0 || piecePosition.y+dy > GameState.BOARD_SIZE-1) ||
            (dy === -1 && !piece.isKing() && piece.getPlayer() === 2) ||
            (dy === 1 && !piece.isKing() && piece.getPlayer() === 1)) {
          continue;
        }
        var destinationPoint = {x: piecePosition.x + dx, y: piecePosition.y + dy};
        var destination = this.pieceAtPosition(destinationPoint);
        if (destination === null) {
          res.push(new Move(piece, piecePosition, destinationPoint));
        }
      }
    }
  };

  GameState.prototype._setPieceAtPosition = function(pos, piece) {
    this._board[pos.x + pos.y*GameState.BOARD_SIZE] = piece;
  };

  function PieceState(id, player, isKing) {
    this._id = id;
    this._player = player;
    this._isKing = false;
  }

  PieceState.prototype.getId = function() {
    return this._id;
  };

  PieceState.prototype.getPlayer = function() {
    return this._player;
  };

  PieceState.prototype.isKing = function() {
    return this._isKing;
  };

  function Move(piece, source, destination) {
    this._piece = piece;
    this._source = source;
    this._destination = destination;
  }

  Move.prototype.getPiece = function(piece) {
    return this._piece;
  };

  Move.prototype.getSource = function() {
    return this._source;
  };

  Move.prototype.getDestination = function() {
    return this._destination;
  };

  Move.prototype.jumpedPosition = function() {
    if (Math.abs(this._destination.x - this._source.x) === 2) {
      return {
        x: this._source.x + (this._destination.x-this._source.x)/2,
        y: this._source.y + (this._destination.y-this._source.y)/2,
      };
    } else {
      return null;
    }
  };

  window.app.GameState = GameState;

})();
