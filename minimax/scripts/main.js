(function() {

  var gameState = new window.app.GameState();
  var boardView = null;

  function initialize() {
    boardView = new window.app.BoardView();
    boardView.on('move', function(id, position) {
      var available = gameState.availableMoves();
      for (var i = 0, len = available.length; i < len; ++i) {
        var move = available[i];
        if (move.getPiece().getId() === id) {
          var dest = move.getDestination();
          if (dest.x === position.x && dest.y === position.y) {
            gameState = gameState.stateAfterMove(move);
            break;
          }
        }
      }
      boardView.updateWithState(gameState);
      if (gameState.playerTurn() === 2) {
        playAITurn();
      }
    });
  }

  function playAITurn() {
    boardView.element().className = 'board-ai-turn';
    window.app.optimalTurn(gameState, function(moves) {
      if (moves.length === 0) {
        handleAILoss();
        return;
      }
      var idx = 0;
      var triggerNext;
      triggerNext = function() {
        gameState = gameState.stateAfterMove(moves[idx++]);
        boardView.updateWithState(gameState);
        if (idx === moves.length) {
          boardView.element().className = '';
          if (gameState.availableMoves().length === 0) {
            handleHumanLoss();
          }
        } else {
          setTimeout(triggerNext, 500);
        }
      };
      setTimeout(triggerNext, 300);
    });
  }

  function handleAILoss() {
    setTimeout(function() {
      alert('The AI has lost!');
    }, 300);
  }

  function handleHumanLoss() {
    setTimeout(function() {
      alert('You have lost!');
    }, 300);
  }

  window.addEventListener('load', initialize);

})();
