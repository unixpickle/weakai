(function() {

  var TIMEOUT_CHECK_DEPTH = 4;

  window.app.optimalTurn = function(gameState, callback) {
    if (gameState.availableMoves().length === 0) {
      callback([]);
      return;
    }

    // Allow a redraw before doing all our computation on the UI thread.
    // TODO: do Minimax in a WebWorker so that we don't need to do this.
    window.requestAnimationFrame(function() {
      callback(minimaxRoot(gameState));
    });
  };

  function minimaxRoot(gameState) {
    var timeout = new Date().getTime() + 1000;
    var depth = 0;
    var bestGuess = [];

    var turns = turnsForState(gameState);
    while (true) {
      ++depth;
      var maxAlternative = -Infinity;
      var minAlternative = Infinity;
      var currentBestGuess = null;
      for (var i = 0, len = turns.length; i < len; ++i) {
        var turn = turns[i];
        try {
          var weight = minimizer(turn.endState, maxAlternative, minAlternative, depth-1, timeout);
          if (weight > maxAlternative) {
            maxAlternative = weight;
            currentBestGuess = turn.moves;
          }
        } catch (e) {
          if (e === 'timeout') {
            return bestGuess;
          } else {
            throw e;
          }
        }
      }
      bestGuess = currentBestGuess;
    }
  }

  function maximizer(gameState, maxAlternative, minAlternative, depthRemaining, timeout) {
    if (depthRemaining === 0) {
      return heuristicForState(gameState);
    } else if (depthRemaining > TIMEOUT_CHECK_DEPTH) {
      if (new Date().getTime() > timeout) {
        throw 'timeout';
      }
    }
    var maximumOutcome = -Infinity;
    var turns = turnsForState(gameState);
    for (var i = 0, len = turns.length; i < len; ++i) {
      var turn = turns[i];
      var value = minimizer(turn.endState, Math.max(maximumOutcome, maxAlternative), minAlternative,
        depthRemaining-1, timeout);
      maximumOutcome = Math.max(maximumOutcome, value);
      if (maximumOutcome >= minAlternative) {
        break;
      }
    }
    return maximumOutcome;
  }

  function minimizer(gameState, maxAlternative, minAlternative, depthRemaining, timeout) {
    if (depthRemaining === 0) {
      return heuristicForState(gameState);
    } else if (depthRemaining > TIMEOUT_CHECK_DEPTH) {
      if (new Date().getTime() > timeout) {
        throw 'timeout';
      }
    }
    var minimumOutcome = Infinity;
    var turns = turnsForState(gameState);
    for (var i = 0, len = turns.length; i < len; ++i) {
      var turn = turns[i];
      var value = maximizer(turn.endState, maxAlternative, Math.min(minimumOutcome, minAlternative),
        depthRemaining-1, timeout);
      minimumOutcome = Math.min(minimumOutcome, value);
      if (minimumOutcome <= maxAlternative) {
        break;
      }
    }
    return minimumOutcome;
  }

  // turnsForState returns all possible turns that the current player could make.
  // A turn is an object with the following keys:
  // - moves: the individual Moves, in order, of which this turn consists.
  // - endState: the GameState that will be reached once all the moves are applied.
  function turnsForState(gameState) {
    var player = gameState.playerTurn();
    var res = [];
    var moves = gameState.availableMoves();
    for (var i = 0, len = moves.length; i < len; ++i) {
      var move = moves[i];
      var newState = gameState.stateAfterMove(move);
      if (newState.playerTurn() !== player) {
        res.push({moves: [move], endState: newState});
      } else {
        var turnContinuations = turnsForState(newState);
        for (var j = 0, len1 = turnContinuations.length; j < len1; ++j) {
          var turn = turnContinuations[j];
          turn.moves.splice(0, 0, move);
          res.push(turn);
        }
      }
    }
    return res;
  }

  function heuristicForState(gameState) {
    var goodness = 0;
    for (var x = 0; x < window.app.GameState.BOARD_SIZE; ++x) {
      for (var y = 0; y < window.app.GameState.BOARD_SIZE; ++y) {
        var piece = gameState.pieceAtPosition({x: x, y: y});
        if (piece === null) {
          continue;
        } else if (piece.getPlayer() === 1) {
          --goodness;
        } else {
          ++goodness;
        }
      }
    }
    return goodness;
  }

})();
